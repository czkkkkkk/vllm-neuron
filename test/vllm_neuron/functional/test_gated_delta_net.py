# SPDX-License-Identifier: Apache-2.0
"""Test the model-agnostic Gated DeltaNet functions."""

import os

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounter

from vllm_neuron.functional import gated_delta_net as gdn

SEED = 42
KERNEL_SIZE = 4
requires_simulator = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="Set NKI_SIMULATOR=1 to run the NKI simulator tests.",
)


def _torch_conv(mixed, weight, state):
    window = weight.shape[-1] - 1
    if state is None:
        padded = F.pad(mixed, (window, 0))
    else:
        padded = torch.cat([state, mixed], dim=-1)
    seq_len = mixed.shape[-1]
    out = padded[:, :, 0:seq_len] * weight[None, :, 0:1]
    for kernel_index in range(1, weight.shape[-1]):
        out = (
            out
            + padded[
                :,
                :,
                kernel_index : kernel_index + seq_len,
            ]
            * weight[None, :, kernel_index : kernel_index + 1]
        )
    return F.silu(out)


def _assert_close(actual, expected, atol=3e-2, rtol=3e-2):
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
    )


def _gdn_inputs(batch, seq_len, num_k_heads, num_v_heads, head_dim):
    torch.manual_seed(SEED)
    query = torch.randn(batch, seq_len, num_k_heads, head_dim).bfloat16()
    key = torch.randn(batch, seq_len, num_k_heads, head_dim).bfloat16()
    value = (torch.randn(batch, seq_len, num_v_heads, head_dim) * 0.5).bfloat16()
    a = torch.randn(batch, seq_len, num_v_heads).bfloat16()
    b = torch.randn(batch, seq_len, num_v_heads).bfloat16()
    A_log = torch.rand(num_v_heads) * 0.5
    dt_bias = torch.rand(num_v_heads)
    state = torch.randn(batch, num_v_heads, head_dim, head_dim) * 0.1
    return query, key, value, a, b, A_log, dt_bias, state


def _gates(a, b, A_log, dt_bias):
    beta = b.float().sigmoid()
    g = -A_log.exp() * F.softplus(a.float() + dt_bias)
    return g, beta


@pytest.mark.parametrize("seq_len", [1, 15, 16, 33])
@pytest.mark.parametrize("with_state", [False, True])
def test_chunked_rule_matches_the_recurrent_rule(seq_len, with_state):
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch=2,
        seq_len=seq_len,
        num_k_heads=2,
        num_v_heads=2,
        head_dim=8,
    )
    g, beta = _gates(a, b, A_log, dt_bias)
    initial_state = state if with_state else None

    actual = gdn.chunked_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
    )
    expected = gdn.recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
    )

    _assert_close(actual[0], expected[0], atol=1e-4, rtol=1e-4)
    _assert_close(actual[1], expected[1], atol=1e-4, rtol=1e-4)


def test_chunked_rule_returns_each_requested_checkpoint():
    batch, seq_len = 3, 32
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch=batch,
        seq_len=seq_len,
        num_k_heads=1,
        num_v_heads=1,
        head_dim=8,
    )
    g, beta = _gates(a, b, A_log, dt_bias)
    offsets = torch.tensor([0, 16, 32])

    _, actual_state = gdn.chunked_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=state,
        checkpoint_offsets=offsets,
    )
    _, checkpoints = gdn.recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=state,
        return_state_checkpoints=True,
    )
    expected_checkpoint = torch.stack(
        (state[0], checkpoints[1, 15], checkpoints[2, 31])
    )

    assert actual_state.shape == (batch, 2, 1, 8, 8)
    _assert_close(actual_state[:, 0], expected_checkpoint, atol=1e-4, rtol=1e-4)
    _assert_close(actual_state[:, 1], checkpoints[:, -1], atol=1e-4, rtol=1e-4)


def test_chunked_rule_returns_intra_chunk_checkpoints():
    batch, seq_len = 3, 32
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch=batch,
        seq_len=seq_len,
        num_k_heads=1,
        num_v_heads=1,
        head_dim=8,
    )
    g, beta = _gates(a, b, A_log, dt_bias)
    offsets = torch.tensor([2, 17, 31])

    _, actual_state = gdn.chunked_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=state,
        checkpoint_offsets=offsets,
    )
    _, checkpoints = gdn.recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=state,
        return_state_checkpoints=True,
    )
    expected_checkpoint = torch.stack(
        (checkpoints[0, 1], checkpoints[1, 16], checkpoints[2, 30])
    )

    _assert_close(actual_state[:, 0], expected_checkpoint, atol=1e-4, rtol=1e-4)
    _assert_close(actual_state[:, 1], checkpoints[:, -1], atol=1e-4, rtol=1e-4)


def test_chunked_rule_compiles_to_one_cpu_graph():
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch=1,
        seq_len=16,
        num_k_heads=1,
        num_v_heads=1,
        head_dim=8,
    )
    gate, beta = _gates(a, b, A_log, dt_bias)
    counter = CompileCounter()

    def step(q, k, v, g, update, initial_state, offsets):
        return gdn.chunked_gated_delta_rule(
            q,
            k,
            v,
            g,
            update,
            initial_state=initial_state,
            checkpoint_offsets=offsets,
        )

    compiled = torch.compile(step, backend=counter, fullgraph=True)
    for offsets in (torch.tensor([0]), torch.tensor([16])):
        compiled(query, key, value, gate, beta, state, offsets)

    assert counter.frame_count == 1


def test_unit_lower_triangular_inverse_matches_torch():
    torch.manual_seed(SEED)
    matrix = torch.randn(2, 3, 16, 16).tril(-1) * 0.05
    actual = gdn.unit_lower_triangular_inverse(matrix, 16)
    eye = torch.eye(16)
    expected = torch.linalg.inv(eye - matrix)

    _assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_decode_helpers_preserve_shapes_and_mask_invalid_rows():
    query = torch.arange(2 * 1 * 1 * 4).reshape(2, 1, 1, 4)
    key = query + 20
    repeated_query, repeated_key = gdn.repeat_gdn_key_heads(
        query,
        key,
        num_v_heads=3,
    )
    a = torch.ones(2, 1, 3)
    b = torch.full_like(a, 2)
    masked_a, masked_b = gdn.mask_gdn_decode_gates(
        a,
        b,
        torch.tensor([True, False]),
    )

    assert repeated_query.shape == (2, 1, 3, 4)
    assert repeated_key.shape == (2, 1, 3, 4)
    assert torch.equal(repeated_query[:, :, 0], query[:, :, 0])
    assert torch.equal(repeated_query[:, :, 2], query[:, :, 0])
    assert torch.equal(repeated_key[:, :, 1], key[:, :, 0])
    assert torch.equal(masked_a[0], a[0])
    assert torch.equal(masked_b[0], b[0])
    assert torch.all(masked_a[1] < -1000)
    assert torch.all(masked_b[1] < -1000)


def test_decode_helpers_reject_invalid_metadata():
    query = torch.zeros(2, 1, 2, 4)

    with pytest.raises(ValueError, match="Provide one value for each request"):
        gdn.mask_gdn_decode_gates(
            torch.zeros(2, 1, 3),
            torch.zeros(2, 1, 3),
            torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="Use a divisible head count"):
        gdn.repeat_gdn_key_heads(query, query, num_v_heads=3)
    with pytest.raises(ValueError, match="Use the same head count"):
        gdn.repeat_gdn_key_heads(query, query[:, :, :1], num_v_heads=2)


def test_decode_gate_mask_follows_the_gate_rank():
    a = torch.ones(3, 4)
    b = torch.full_like(a, 2)

    masked_a, masked_b = gdn.mask_gdn_decode_gates(
        a, b, torch.tensor([True, False, True])
    )

    assert masked_a.shape == a.shape
    assert torch.equal(masked_a[0], a[0])
    assert torch.equal(masked_b[2], b[2])
    assert torch.all(masked_a[1] < -1000)
    assert torch.all(masked_b[1] < -1000)


@pytest.mark.parametrize("shape", [(8, 1, 4), (4, 4)])
def test_conv1d_rejects_a_weight_without_channel_and_kernel_axes(shape):
    mixed = torch.zeros(1, 8, 3, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="Use a \\[channels, kernel\\] weight"):
        gdn.gdn_conv1d(mixed, torch.zeros(shape, dtype=torch.bfloat16), None)


def test_prefill_checkpoint_masks_the_suffix(monkeypatch):
    class FakeKernel:
        def __init__(self):
            self.calls = []

        def __getitem__(self, _grid):
            return self

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            value = kwargs["value"]
            state = kwargs["recurrent_state"]
            state = state + kwargs["beta"].sum(dim=-1)[..., None, None]
            return value, state

    kernel = FakeKernel()
    monkeypatch.setattr(gdn, "wrapped_gdn_prefill", kernel)
    monkeypatch.setattr(gdn, "get_logical_nc_config", lambda: 1)

    batch, seq_len, num_heads, head_dim = 3, 4, 1, 2
    query = torch.randn(batch, seq_len, num_heads, head_dim).bfloat16()
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = torch.ones(batch, seq_len, num_heads)
    beta = torch.ones(batch, seq_len, num_heads)
    state = torch.zeros(batch, num_heads, head_dim, head_dim)
    offsets = torch.tensor([0, 2, seq_len])

    _, actual_state = gdn.gdn_prefill(
        query,
        key,
        value,
        gate,
        beta,
        state,
        torch.empty(0),
        checkpoint_offsets=offsets,
    )

    checkpoint_call = kernel.calls[1]
    expected_mask = torch.tensor(
        [
            [[False, False, False, False]],
            [[True, True, False, False]],
            [[True, True, True, True]],
        ]
    )
    assert len(kernel.calls) == 2
    assert torch.equal(checkpoint_call["beta"].bool(), expected_mask)
    assert torch.equal(checkpoint_call["g_log"].bool(), expected_mask)
    assert torch.equal(actual_state[:, 0, 0, 0, 0], torch.tensor([0.0, 2.0, 4.0]))
    assert torch.equal(actual_state[:, 1, 0, 0, 0], torch.tensor([4.0, 4.0, 4.0]))


@requires_simulator
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("seq_len", [1, 16])
def test_conv1d_matches_the_torch_convolution(seq_len, with_state):
    if seq_len == 1 and not with_state:
        pytest.skip("A decode step requires a saved window.")
    torch.manual_seed(SEED)
    batch, conv_dim = 2, 192
    mixed = torch.randn(batch, conv_dim, seq_len).bfloat16()
    weight = (torch.randn(conv_dim, KERNEL_SIZE) * 0.5).bfloat16()
    state = (
        torch.randn(batch, conv_dim, KERNEL_SIZE - 1).bfloat16() if with_state else None
    )

    actual = gdn.gdn_conv1d(mixed, weight, state)
    expected = _torch_conv(
        mixed.float(),
        weight.float(),
        None if state is None else state.float(),
    )

    _assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@requires_simulator
def test_conv1d_runs_the_8192_token_prefill_bucket():
    torch.manual_seed(SEED)
    seq_len, conv_dim = 8192, 128
    mixed = torch.randn(1, conv_dim, seq_len).bfloat16()
    weight = (torch.randn(conv_dim, KERNEL_SIZE) * 0.5).bfloat16()
    state = torch.randn(1, conv_dim, KERNEL_SIZE - 1).bfloat16()

    actual = gdn.gdn_conv1d(mixed, weight, state)
    expected = _torch_conv(mixed.float(), weight.float(), state.float())

    assert actual.shape == (1, conv_dim, seq_len)
    _assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_conv1d_uses_the_target_default_lnc(monkeypatch):
    class CallOnlyKernel:
        def __getitem__(self, _grid):
            raise AssertionError("The GDN convolution must not set a fixed grid.")

        def __call__(self, **kwargs):
            return kwargs.get("x", kwargs.get("mixed")), None

    monkeypatch.setattr(gdn, "wrapped_gdn_conv1d_prefill", CallOnlyKernel())
    mixed = torch.randn(2, 192, 16).bfloat16()
    weight = torch.randn(192, KERNEL_SIZE).bfloat16()
    state = torch.randn(2, 192, KERNEL_SIZE - 1).bfloat16()

    actual = gdn.gdn_conv1d(mixed, weight, state)

    assert torch.equal(actual, mixed)


@requires_simulator
def test_prefill_matches_the_chunked_reference():
    batch, seq_len, num_k_heads, num_v_heads, head_dim = 1, 128, 1, 2, 32
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch,
        seq_len,
        num_k_heads,
        num_v_heads,
        head_dim,
    )
    gate, beta = _gates(a, b, A_log, dt_bias)
    saved_state = state.clone()

    actual_out, actual_state = gdn.gdn_prefill(
        query,
        key,
        value,
        gate,
        beta,
        state,
        gdn.build_gdn_mask_pack(),
    )
    repeat = num_v_heads // num_k_heads
    expected_out, expected_state = gdn.chunked_gated_delta_rule(
        query.float().repeat_interleave(repeat, dim=2),
        key.float().repeat_interleave(repeat, dim=2),
        value.float(),
        g=gate,
        beta=beta,
        initial_state=state,
    )

    _assert_close(actual_out, expected_out)
    _assert_close(actual_state, expected_state)
    assert torch.equal(state, saved_state)


@requires_simulator
@pytest.mark.parametrize("with_state", [False, True])
def test_prefill_checkpoint_matches_the_chunked_reference(with_state):
    batch, seq_len, num_k_heads, num_v_heads, head_dim = 3, 128, 1, 2, 32
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch,
        seq_len,
        num_k_heads,
        num_v_heads,
        head_dim,
    )
    gate, beta = _gates(a, b, A_log, dt_bias)
    initial_state = state if with_state else None
    offsets = torch.tensor([0, 64, seq_len])

    actual = gdn.gdn_prefill(
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
        gdn.build_gdn_mask_pack(),
        checkpoint_offsets=offsets,
    )
    repeat = num_v_heads // num_k_heads
    expected = gdn.chunked_gated_delta_rule(
        query.float().repeat_interleave(repeat, dim=2),
        key.float().repeat_interleave(repeat, dim=2),
        value.float(),
        g=gate,
        beta=beta,
        initial_state=initial_state,
        checkpoint_offsets=offsets,
    )

    _assert_close(actual[0], expected[0])
    _assert_close(actual[1], expected[1])


def _gated_rms_norm(x, z, weight, eps=1e-6):
    normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return normalized * weight * F.silu(z)


@requires_simulator
def test_decode_matches_the_recurrent_reference():
    batch, num_k_heads, num_v_heads, head_dim = 2, 1, 2, 32
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch,
        1,
        num_k_heads,
        num_v_heads,
        head_dim,
    )
    z = torch.randn(batch, 1, num_v_heads * head_dim).bfloat16()
    norm_weight = (1.0 + 0.1 * torch.randn(head_dim)).bfloat16()

    actual_out, actual_state = gdn.gdn_decode(
        query,
        key,
        value,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state,
    )
    gate, beta = _gates(a, b, A_log, dt_bias)
    repeat = num_v_heads // num_k_heads
    core_out, expected_state = gdn.recurrent_gated_delta_rule(
        query.float().repeat_interleave(repeat, dim=2),
        key.float().repeat_interleave(repeat, dim=2),
        value.float(),
        g=gate,
        beta=beta,
        initial_state=state,
    )
    expected_out = _gated_rms_norm(
        core_out.reshape(-1, head_dim).bfloat16(),
        z.reshape(-1, head_dim),
        norm_weight,
    ).reshape(batch, 1, -1)

    _assert_close(actual_out, expected_out)
    _assert_close(actual_state, expected_state, atol=2e-2, rtol=2e-2)


@requires_simulator
def test_decode_tracks_the_fp32_rule_over_many_steps():
    """Feed the kernel state back for 64 steps and bound the drift.

    The state must stay close to the FP32 rule. A single output element can
    differ by a few BF16 steps on any step, with or without state feedback, so
    bound the output with its relative L2 error.
    """
    batch, num_k_heads, num_v_heads, head_dim, steps = 2, 1, 2, 32, 64
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch,
        steps,
        num_k_heads,
        num_v_heads,
        head_dim,
    )
    z = torch.randn(batch, steps, num_v_heads * head_dim).bfloat16()
    norm_weight = (1.0 + 0.1 * torch.randn(head_dim)).bfloat16()
    gate, beta = _gates(a, b, A_log, dt_bias)
    repeat = num_v_heads // num_k_heads
    core_out, expected_state = gdn.recurrent_gated_delta_rule(
        query.float().repeat_interleave(repeat, dim=2),
        key.float().repeat_interleave(repeat, dim=2),
        value.float(),
        g=gate,
        beta=beta,
        initial_state=state,
    )

    actual_state = state
    for step in range(steps):
        token = slice(step, step + 1)
        actual_out, actual_state = gdn.gdn_decode(
            query[:, token],
            key[:, token],
            value[:, token],
            a[:, token],
            b[:, token],
            z[:, token],
            A_log,
            dt_bias,
            norm_weight,
            actual_state,
        )
        expected_out = _gated_rms_norm(
            core_out[:, token].reshape(-1, head_dim).bfloat16(),
            z[:, token].reshape(-1, head_dim),
            norm_weight,
        ).reshape(batch, 1, -1)
        output_error = (actual_out.float() - expected_out.float()).norm()
        assert output_error <= 5e-2 * expected_out.float().norm(), step

    _assert_close(actual_state, expected_state, atol=5e-3, rtol=5e-3)


@requires_simulator
def test_decode_uses_one_program_for_one_batch_head(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    batch, num_k_heads, num_v_heads, head_dim = 1, 1, 1, 16
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch,
        1,
        num_k_heads,
        num_v_heads,
        head_dim,
    )
    z = torch.randn(batch, 1, num_v_heads * head_dim).bfloat16()
    norm_weight = (1.0 + 0.1 * torch.randn(head_dim)).bfloat16()

    actual_out, actual_state = gdn.gdn_decode(
        query,
        key,
        value,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state,
    )
    gate, beta = _gates(a, b, A_log, dt_bias)
    core_out, expected_state = gdn.recurrent_gated_delta_rule(
        query.float(),
        key.float(),
        value.float(),
        g=gate,
        beta=beta,
        initial_state=state,
    )
    expected_out = _gated_rms_norm(
        core_out.reshape(-1, head_dim).bfloat16(),
        z.reshape(-1, head_dim),
        norm_weight,
    ).reshape(batch, 1, -1)

    _assert_close(actual_out, expected_out)
    _assert_close(actual_state, expected_state, atol=2e-2, rtol=2e-2)


@requires_simulator
def test_decode_leaves_an_invalid_row_state_unchanged():
    batch, num_k_heads, num_v_heads, head_dim = 2, 1, 2, 32
    query, key, value, a, b, A_log, dt_bias, state = _gdn_inputs(
        batch,
        1,
        num_k_heads,
        num_v_heads,
        head_dim,
    )
    z = torch.randn(batch, 1, num_v_heads * head_dim).bfloat16()
    norm_weight = torch.ones(head_dim, dtype=torch.bfloat16)

    _, new_state = gdn.gdn_decode(
        query,
        key,
        value,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state,
        torch.tensor([[True], [False]]),
    )

    _assert_close(new_state[1], state[1], atol=1e-6, rtol=0)
    assert not torch.allclose(new_state[0], state[0])


@pytest.fixture
def kernels_enabled(monkeypatch):
    monkeypatch.setattr(gdn, "can_run_kernel", lambda *_: True)


def test_reject_reasons_name_the_unsupported_shape(kernels_enabled):
    query = torch.zeros(1, 100, 2, 128, dtype=torch.bfloat16)
    value = torch.zeros(1, 100, 4, 128, dtype=torch.bfloat16)
    assert "multiple of 128" in gdn.gdn_prefill_reject_reason(query, value)
    assert "not one token" in gdn.gdn_decode_reject_reason(query, value, 1e-6)

    query_one, value_one = query[:, :1], value[:, :1]
    assert gdn.gdn_decode_reject_reason(query_one, value_one, 1e-6) is None
    assert "eps" in gdn.gdn_decode_reject_reason(query_one, value_one, 1e-5)
    assert "bfloat16" in gdn.gdn_decode_reject_reason(
        query_one.float(),
        value_one,
        1e-6,
    )

    wrong_state = torch.zeros(1, 256, 4, dtype=torch.bfloat16)
    mixed = torch.zeros(1, 256, 16, dtype=torch.bfloat16)
    assert "conv_state holds 4" in gdn.gdn_conv1d_reject_reason(
        mixed,
        wrong_state,
        4,
    )
    bucket = torch.zeros(1, 256, 8192, dtype=torch.bfloat16)
    assert gdn.gdn_conv1d_reject_reason(bucket, None, 4) is None
    long_input = torch.zeros(1, 256, 8193, dtype=torch.bfloat16)
    assert "larger than" in gdn.gdn_conv1d_reject_reason(long_input, None, 4)


def test_reject_reasons_report_disabled_kernels(monkeypatch):
    monkeypatch.setattr(gdn, "can_run_kernel", lambda *_: False)
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    value = torch.zeros(1, 128, 4, 128, dtype=torch.bfloat16)
    reasons = (
        gdn.gdn_prefill_reject_reason(query, value),
        gdn.gdn_decode_reject_reason(query[:, :1], value[:, :1], 1e-6),
        gdn.gdn_conv1d_reject_reason(
            torch.zeros(1, 256, 16, dtype=torch.bfloat16),
            None,
            4,
        ),
    )

    assert all(reason is not None and "disabled" in reason for reason in reasons)
