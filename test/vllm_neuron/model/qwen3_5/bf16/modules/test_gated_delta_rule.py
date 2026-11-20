# SPDX-License-Identifier: Apache-2.0
"""Gated delta rule tests for the Qwen3.8 hybrid model.

Validates that chunked_gated_delta_rule matches recurrent_gated_delta_rule,
also from a saved state, and that unit_lower_triangular_inverse is exact.

The model holds two forms of the same recurrence. The sequential form steps one
token at a time, and it is the reference. The chunked form does the work of a
chunk with matrix products, and it keeps the graph small on a long prefill.

These tests pin the two forms together. They need no checkpoint and no device.

The tests read no checkpoint and need no device, so ``make test`` runs them. The
head dimensions stay small on purpose. The recurrence does not depend on the head
width, so a small width keeps the build-time pass quick.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        test/vllm_neuron/model/qwen3_5/bf16/modules/test_gated_delta_rule.py -v
"""

import pytest
import torch

from vllm_neuron.functional.gated_delta_net import (
    chunked_gated_delta_rule,
    recurrent_gated_delta_rule,
    unit_lower_triangular_inverse,
)

pytestmark = [pytest.mark.fast]

# The real model uses 128 for both. The math is independent of the width.
HEAD_K_DIM = 16
HEAD_V_DIM = 16


def make_inputs(batch: int, seq_len: int, num_v_heads: int, seed: int = 42):
    """Build one random input set for the delta rule.

    The decay must stay at or below zero, so the state never grows.
    """
    torch.manual_seed(seed)
    query = torch.randn(batch, seq_len, num_v_heads, HEAD_K_DIM)
    key = torch.randn(batch, seq_len, num_v_heads, HEAD_K_DIM)
    value = torch.randn(batch, seq_len, num_v_heads, HEAD_V_DIM)
    beta = torch.rand(batch, seq_len, num_v_heads)
    g = -torch.rand(batch, seq_len, num_v_heads) * 0.5
    return query, key, value, g, beta


def make_realistic_system(chunk_size: int, heads: int = 4, head_k_dim: int = 16):
    """Build the matrix that the chunked recurrence passes to the inverse.

    ``chunked_gated_delta_rule`` forms ``-(k_beta @ key^T * pairwise_decay)``
    and keeps the strictly lower triangle. The keys carry an L2 norm, beta stays
    inside ``(0, 1)``, and the pairwise decay stays at or below one, so each
    entry stays well below one.

    A matrix of plain normal samples is a much harder problem. At width 64 its
    inverse holds entries near 1e7, so an absolute tolerance then measures the
    float64 limit and not the implementation. Build the realistic matrix
    instead, and keep the normal samples for a relative check.
    """
    key = torch.nn.functional.normalize(
        torch.randn(heads, chunk_size, head_k_dim, dtype=torch.float64), dim=-1
    )
    beta = torch.rand(heads, chunk_size, 1, dtype=torch.float64)
    decay = (-torch.rand(heads, chunk_size, dtype=torch.float64) * 0.5).cumsum(-1)
    strictly_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool).triu(1)
    pairwise = (
        (decay.unsqueeze(2) - decay.unsqueeze(1))
        .masked_fill(strictly_upper, float("-inf"))
        .exp()
    )
    return (-((key * beta) @ key.transpose(-1, -2) * pairwise)).tril(-1)


@pytest.mark.parametrize("chunk_size", [4, 8, 16, 64])
def test_triangular_inverse_inverts_a_realistic_system(chunk_size):
    """The product form must invert ``I - a`` in the range that the model uses.

    The identity holds exact zeros, so compare against it with an absolute
    tolerance. A realistic ``a`` keeps the result near one, so the tolerance
    stays tight.
    """
    torch.manual_seed(0)
    a = make_realistic_system(chunk_size)
    eye = torch.eye(chunk_size, dtype=torch.float64)

    product = unit_lower_triangular_inverse(a, chunk_size) @ (eye - a)

    assert a.abs().max() < 1.0, "the realistic system must stay below one"
    torch.testing.assert_close(product, eye.expand_as(product), rtol=0, atol=1e-12)


@pytest.mark.parametrize("chunk_size", [4, 8, 16, 64])
def test_triangular_inverse_matches_the_explicit_series(chunk_size):
    """The product form must equal ``I + a + a^2 + ...`` term by term.

    Plain normal samples give a hard problem on purpose. The entries of the
    result then span many orders of magnitude, so compare relatively. An
    absolute tolerance would measure the float64 limit of the reference.
    """
    torch.manual_seed(1)
    a = torch.randn(2, chunk_size, chunk_size, dtype=torch.float64).tril(-1)
    eye = torch.eye(chunk_size, dtype=torch.float64)

    series = eye.expand_as(a).clone()
    power = eye.expand_as(a).clone()
    for _ in range(chunk_size - 1):
        power = power @ a
        series = series + power

    torch.testing.assert_close(
        unit_lower_triangular_inverse(a, chunk_size), series, rtol=1e-9, atol=0
    )


@pytest.mark.parametrize("chunk_size", [4, 8, 16, 64])
def test_triangular_inverse_is_near_exact_on_a_realistic_system(chunk_size):
    """The realistic system must give a result at the float64 limit.

    This is the operating range of the model, so the error must be tiny. The
    measured error at width 64 is about 6e-17.
    """
    torch.manual_seed(2)
    a = make_realistic_system(chunk_size)
    eye = torch.eye(chunk_size, dtype=torch.float64)

    series = eye.expand_as(a).clone()
    power = eye.expand_as(a).clone()
    for _ in range(chunk_size - 1):
        power = power @ a
        series = series + power

    torch.testing.assert_close(
        unit_lower_triangular_inverse(a, chunk_size), series, rtol=0, atol=1e-13
    )


@pytest.mark.parametrize(
    "batch,seq_len,num_v_heads",
    [(1, 1, 4), (1, 8, 4), (2, 64, 4), (1, 65, 4), (2, 130, 6)],
)
def test_chunked_matches_sequential(batch, seq_len, num_v_heads):
    """The chunked form must match the sequential form, output and state.

    A sequence length that is not a whole number of chunks exercises the pad
    path. A padded token must change neither the output nor the state.
    """
    inputs = make_inputs(batch, seq_len, num_v_heads)

    out_ref, state_ref = recurrent_gated_delta_rule(*inputs)
    out_chunked, state_chunked = chunked_gated_delta_rule(*inputs)

    assert out_chunked.shape == out_ref.shape
    assert state_chunked.shape == state_ref.shape
    torch.testing.assert_close(out_chunked, out_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(state_chunked, state_ref, rtol=2e-4, atol=2e-4)


def test_default_chunk_keeps_correlated_keys_bounded():
    """A repeated-key prefill must not magnify a bounded recurrence.

    This is the numerical corner that a wide Neumann solve misses: its large
    intermediate powers need cancellation that float32 cannot preserve. The
    16-token production chunk stays close to the token-by-token reference.
    """
    batch, seq_len, num_v_heads = 1, 128, 1
    query = torch.ones(batch, seq_len, num_v_heads, HEAD_K_DIM)
    key = torch.ones_like(query)
    torch.manual_seed(42)
    value = torch.randn(batch, seq_len, num_v_heads, HEAD_V_DIM)
    g = torch.full(
        (batch, seq_len, num_v_heads),
        -float(torch.nn.functional.softplus(torch.tensor(-8.0))),
    )
    beta = torch.full(
        (batch, seq_len, num_v_heads),
        float(torch.sigmoid(torch.tensor(3.0))),
    )

    out_ref, state_ref = recurrent_gated_delta_rule(query, key, value, g, beta)
    out_chunked, state_chunked = chunked_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
    )

    assert state_chunked.abs().max() < 1
    torch.testing.assert_close(
        state_chunked,
        state_ref,
        rtol=2e-3,
        atol=5e-4,
    )
    torch.testing.assert_close(
        out_chunked,
        out_ref,
        rtol=2e-3,
        atol=5e-4,
    )


def test_chunked_continues_from_a_state():
    """Two chunked calls must equal one call over the joined sequence.

    This is the property that a segmented prefill needs.
    """
    batch, num_v_heads, first, second = 1, 4, 64, 32
    query, key, value, g, beta = make_inputs(batch, first + second, num_v_heads)

    out_whole, state_whole = chunked_gated_delta_rule(query, key, value, g, beta)

    def head(x):
        return x[:, :first]

    def tail(x):
        return x[:, first:]

    out_a, state_a = chunked_gated_delta_rule(
        head(query), head(key), head(value), head(g), head(beta)
    )
    out_b, state_b = chunked_gated_delta_rule(
        tail(query),
        tail(key),
        tail(value),
        tail(g),
        tail(beta),
        initial_state=state_a,
    )

    torch.testing.assert_close(
        torch.cat([out_a, out_b], dim=1), out_whole, rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(state_b, state_whole, rtol=2e-4, atol=2e-4)


def test_sequential_rule_returns_each_state_checkpoint():
    """Each checkpoint must equal a run that stops at that token."""
    batch, seq_len, num_v_heads = 2, 4, 3
    inputs = make_inputs(batch, seq_len, num_v_heads)
    initial_state = torch.randn(
        batch,
        num_v_heads,
        HEAD_K_DIM,
        HEAD_V_DIM,
    )

    output, checkpoints = recurrent_gated_delta_rule(
        *inputs,
        initial_state=initial_state,
        return_state_checkpoints=True,
    )

    assert output.shape == (batch, seq_len, num_v_heads, HEAD_V_DIM)
    assert checkpoints.shape == (
        batch,
        seq_len,
        num_v_heads,
        HEAD_K_DIM,
        HEAD_V_DIM,
    )
    for position in range(seq_len):
        prefix = tuple(value[:, : position + 1] for value in inputs)
        _, expected = recurrent_gated_delta_rule(
            *prefix,
            initial_state=initial_state,
        )
        torch.testing.assert_close(checkpoints[:, position], expected)


def test_zero_beta_leaves_the_state_unchanged():
    """A token with ``beta = 0`` writes nothing, so only the decay applies."""
    batch, seq_len, num_v_heads = 1, 8, 4
    query, key, value, g, beta = make_inputs(batch, seq_len, num_v_heads)
    beta = torch.zeros_like(beta)
    start = torch.randn(batch, num_v_heads, HEAD_K_DIM, HEAD_V_DIM)

    _, state = chunked_gated_delta_rule(query, key, value, g, beta, initial_state=start)

    # g has shape [batch, seq, num_v_heads]. The sum over the sequence gives the
    # log of the product of the decays of each head.
    total_decay = g.sum(dim=1)[..., None, None]
    torch.testing.assert_close(state, start * total_decay.exp(), rtol=2e-4, atol=2e-4)
