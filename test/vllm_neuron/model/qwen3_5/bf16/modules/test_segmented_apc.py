# SPDX-License-Identifier: Apache-2.0
"""Test Qwen3.8 segmented prefill and aligned recurrent checkpoints."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaAttentionBackendEnum, MambaSpec

import vllm_neuron.model.qwen3_5.model_bf16 as model_bf16
from vllm_neuron.model.kv_cache import RecurrentLayerSpec, carve_mamba_state_views
from vllm_neuron.model.qwen3_5.config import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen3_5TextConfig,
)

pytestmark = [pytest.mark.fast]


def _config(layer_types: list[str]) -> Qwen3_5TextConfig:
    """Create a small Qwen3.8 configuration."""
    return Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=layer_types,
        mrope_section=[1, 0, 0],
        torch_dtype=torch.float32,
    )


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use one tensor-parallel rank in each test."""
    group = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        device_group=None,
        all_reduce=lambda value: value,
        all_gather=lambda value, dim: value,
    )
    monkeypatch.setattr(model_bf16, "get_tp_group", lambda: group)


def _state_views(
    layer_spec: model_bf16.LayerSpec,
    num_pages: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Allocate recurrent pages for one GDN layer."""
    assert layer_spec.state_shapes is not None
    assert layer_spec.state_dtypes is not None
    page_size = sum(
        torch.empty(shape, dtype=dtype).numel()
        * torch.empty((), dtype=dtype).element_size()
        for shape, dtype in zip(
            layer_spec.state_shapes,
            layer_spec.state_dtypes,
            strict=True,
        )
    )
    spec = MambaSpec(
        block_size=4,
        shapes=layer_spec.state_shapes,
        dtypes=layer_spec.state_dtypes,
        page_size_padded=page_size,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1,
    )
    raw = torch.zeros(num_pages * page_size, dtype=torch.int8)
    return raw, carve_mamba_state_views(raw, spec)


def test_cache_spec_declares_one_aligned_prefill_checkpoint() -> None:
    """Declare the checkpoint count and token alignment for each GDN layer."""
    spec = model_bf16.build_qwen3_5_kv_spec(Qwen3_5TextConfig(), tp_size=4)

    recurrent = next(
        layer for layer in spec.layers if isinstance(layer, RecurrentLayerSpec)
    )

    assert recurrent.state_num_prefill_checkpoint_blocks == 1
    assert recurrent.state_prefill_checkpoint_alignment == 64


def test_segmented_full_attention_reuses_the_first_chunk() -> None:
    """Match one full attention call with two cache-backed chunks."""
    torch.manual_seed(44)
    config = _config([FULL_ATTENTION])
    segmented = model_bf16.Qwen3_5Attention(config, layer_idx=0)
    reference = model_bf16.Qwen3_5Attention(config, layer_idx=0)
    for parameter in segmented.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)
    reference.load_state_dict(segmented.state_dict())

    block_size = 2
    segmented.k_cache = torch.zeros(3, 1, block_size, config.head_dim)
    segmented.v_cache = torch.zeros_like(segmented.k_cache)
    hidden = torch.randn(1, 4, config.hidden_size)
    positions = torch.arange(4).unsqueeze(0)
    rotary = model_bf16.Qwen3_5RotaryEmbedding(config)
    rope = rotary(positions, hidden.dtype)
    expected = reference(
        hidden,
        rope,
        model_bf16.build_causal_mask(4, 4, hidden.dtype, hidden),
    )

    chunks = []
    for start in (0, 2):
        chunk = hidden[:, start : start + 2]
        chunks.append(
            segmented(
                chunk,
                (
                    rope[0][:, start : start + 2],
                    rope[1][:, start : start + 2],
                ),
                model_bf16.build_causal_mask(
                    2,
                    2,
                    hidden.dtype,
                    chunk,
                ),
                {
                    "slot_mapping": torch.arange(2 + start, 4 + start),
                    "block_size": block_size,
                    "block_table_tensor": torch.tensor(
                        [[1, 2]],
                        dtype=torch.int32,
                    ),
                    "cached_seq_len": torch.tensor([[start]], dtype=torch.int32),
                    "kv_segment_size": 512,
                },
                positions[:, start : start + 2],
            )
        )

    torch.testing.assert_close(torch.cat(chunks, dim=1), expected)


def test_segmented_attention_requires_a_tensor_cache_length() -> None:
    """Reject segmented metadata without a tensor cache length."""
    config = _config([FULL_ATTENTION])
    attention = model_bf16.Qwen3_5Attention(config, layer_idx=0)
    attention.k_cache = torch.zeros(2, 1, 2, config.head_dim)
    attention.v_cache = torch.zeros_like(attention.k_cache)
    hidden = torch.randn(1, 2, config.hidden_size)
    positions = torch.tensor([[0, 1]])
    rope = model_bf16.Qwen3_5RotaryEmbedding(config)(positions, hidden.dtype)

    with pytest.raises(TypeError, match="cache length is not a tensor"):
        attention(
            hidden,
            rope,
            model_bf16.build_causal_mask(2, 2, hidden.dtype, hidden),
            {
                "slot_mapping": torch.tensor([2, 3]),
                "block_size": 2,
                "block_table_tensor": torch.tensor([[1]], dtype=torch.int32),
                "kv_segment_size": 512,
            },
            positions,
        )


def test_aligned_prefill_writes_checkpoint_and_final_state() -> None:
    """Write the internal checkpoint and final state without changing the prefix."""
    torch.manual_seed(43)
    config = _config([LINEAR_ATTENTION])
    paged = model_bf16.Qwen3_5Model(config)
    reference = model_bf16.Qwen3_5Model(config)
    for parameter in paged.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)
    reference.load_state_dict(paged.state_dict())

    layer_spec = model_bf16.build_qwen3_5_kv_spec(config, tp_size=1).layers[0]
    raw, views = _state_views(layer_spec, num_pages=4)
    views[0][1].normal_(mean=0.0, std=0.05)
    views[1][1].normal_(mean=0.0, std=0.05)
    prefix_before = raw.clone()
    bases, layout = model_bf16.prepare_mamba_state_cache_access(views)
    mixer = paged.layers[0].linear_attn
    mixer.state_cache_bases = tuple(bases)
    mixer.state_cache_layout = layout
    mixer.state_cache_group = 0

    initial_state = (
        views[0][1].transpose(-1, -2).unsqueeze(0).clone(),
        views[1][1].transpose(-1, -2).unsqueeze(0).clone(),
    )
    input_ids = torch.arange(128).remainder(config.vocab_size).unsqueeze(0)
    positions = torch.arange(128).unsqueeze(0)
    checkpoint_offsets = torch.tensor([64], dtype=torch.int32)
    expected, states = reference(
        input_ids,
        positions,
        linear_states={0: initial_state},
        return_linear_states=True,
        checkpoint_offsets=checkpoint_offsets,
    )
    actual = paged(
        input_ids,
        positions,
        {
            "layers.0.linear_attn": {
                "block_table_tensor": torch.tensor([[1, 2, 3]], dtype=torch.int32),
                "recurrent_state_read_page_ids": torch.tensor(
                    [[1]],
                    dtype=torch.int32,
                ),
                "recurrent_state_write_page_ids": torch.tensor(
                    [[2, 3]],
                    dtype=torch.int32,
                ),
                "recurrent_state_write_valid_mask": torch.tensor(
                    [[True, True]],
                ),
                "recurrent_state_cold": torch.tensor([False]),
                "recurrent_state_prefill_checkpoint_offsets": checkpoint_offsets,
            }
        },
    )

    expected_conv, expected_recurrent = states[0]
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        raw[: raw.numel() // 2], prefix_before[: raw.numel() // 2]
    )
    torch.testing.assert_close(views[0][2], expected_conv[0, 0].transpose(-1, -2))
    torch.testing.assert_close(views[0][3], expected_conv[0, 1].transpose(-1, -2))
    torch.testing.assert_close(
        views[1][2],
        expected_recurrent[0, 0].transpose(-1, -2),
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        views[1][3],
        expected_recurrent[0, 1].transpose(-1, -2),
        rtol=2e-4,
        atol=2e-4,
    )


def test_model_rejects_a_non_tensor_checkpoint_offset() -> None:
    """Reject a cache checkpoint offset that is not a tensor."""
    config = _config([LINEAR_ATTENTION])
    model = model_bf16.Qwen3_5Model(config)

    with pytest.raises(TypeError, match="checkpoint offsets are not a tensor"):
        model(
            torch.tensor([[1]]),
            torch.tensor([[0]]),
            {
                "layers.0.linear_attn": {
                    "block_table_tensor": torch.tensor([[1]], dtype=torch.int32),
                    "recurrent_state_read_page_ids": torch.tensor(
                        [[1]],
                        dtype=torch.int32,
                    ),
                    "recurrent_state_write_page_ids": torch.tensor(
                        [[1]],
                        dtype=torch.int32,
                    ),
                    "recurrent_state_prefill_checkpoint_offsets": [0],
                }
            },
            return_linear_states=True,
        )
