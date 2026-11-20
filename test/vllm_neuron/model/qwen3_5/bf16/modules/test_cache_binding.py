# SPDX-License-Identifier: Apache-2.0
"""Test Qwen3.8 BF16 cache binding."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.v1.kv_cache_interface import (
    MambaAttentionBackendEnum,
    MambaSpec,
)

import vllm_neuron.model.qwen3_5.model_bf16 as model_bf16
from vllm_neuron.model.kv_cache import RecurrentLayerSpec, carve_mamba_state_views
from vllm_neuron.model.qwen3_5.config import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen3_5TextConfig,
)

pytestmark = [pytest.mark.fast]


def _config(layer_types: list[str]) -> Qwen3_5TextConfig:
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
def _single_rank(monkeypatch):
    group = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        device_group=None,
        all_reduce=lambda value: value,
        all_gather=lambda value, dim: value,
    )
    monkeypatch.setattr(model_bf16, "get_tp_group", lambda: group)


def _state_views(layer_spec, num_pages: int = 4):
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
        block_size=1,
        shapes=layer_spec.state_shapes,
        dtypes=layer_spec.state_dtypes,
        page_size_padded=page_size,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="none",
        num_heads=1,
        tokens_per_state=-1,
        tp_replicated=False,
    )
    raw = torch.zeros(num_pages * page_size, dtype=torch.int8)
    return raw, carve_mamba_state_views(raw, spec)


def test_cache_spec_declares_attention_and_recurrent_geometry():
    spec = model_bf16.build_qwen3_5_kv_spec(Qwen3_5TextConfig(), tp_size=4)

    recurrent = [
        layer for layer in spec.layers if isinstance(layer, RecurrentLayerSpec)
    ]
    attention = [
        layer for layer in spec.layers if not isinstance(layer, RecurrentLayerSpec)
    ]

    assert len(recurrent) == 48
    assert len(attention) == 16
    assert recurrent[0].name == "layers.0.linear_attn"
    assert recurrent[0].state_shapes == ((3, 2560), (12, 128, 128))
    assert recurrent[0].state_dtypes == (torch.bfloat16, torch.float32)
    assert recurrent[0].state_names == ("conv", "recurrent")
    assert recurrent[0].state_backend_type is MambaAttentionBackendEnum.GDN_ATTN
    assert attention[0].name == "layers.3.self_attn"
    assert attention[0].num_kv_heads == 1
    assert attention[0].head_size == 256


def test_cache_spec_preserves_the_upstream_recurrent_axis_order():
    config = _config([LINEAR_ATTENTION])
    config.linear_key_head_dim = 2

    recurrent = model_bf16.build_qwen3_5_kv_spec(config, tp_size=2).layers[0]

    assert recurrent.state_shapes == ((3, 12), (2, 4, 2))


def test_recurrent_token_mask_detects_zero_padded_positions():
    # Model Runner V2 pads a short row with position zero instead of repeating
    # its last real position.
    positions = torch.tensor([[7, 8, 9, 0, 0], [0, 1, 0, 0, 0]])
    state_write_valid = torch.tensor([[True], [True]])

    actual = model_bf16.build_recurrent_token_mask(positions, state_write_valid)

    assert actual.tolist() == [
        [True, True, True, False, False],
        [True, True, False, False, False],
    ]


def test_recurrent_token_mask_marks_real_prefix_and_active_rows():
    positions = torch.tensor(
        [
            [0, 1, 2, 2, 2],
            [100, 101, 102, 103, 103],
            [0, 0, 0, 0, 0],
        ]
    )
    state_write_valid = torch.tensor([[True], [True], [False]])

    actual = model_bf16.build_recurrent_token_mask(
        positions,
        state_write_valid,
    )

    expected = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "positions, state_write_valid, message",
    [
        (
            torch.zeros(4),
            torch.ones(1, 1),
            "position IDs must have shape",
        ),
        (
            torch.zeros(1, 4),
            torch.ones(1),
            "write-valid mask must have shape",
        ),
        (
            torch.zeros(2, 4),
            torch.ones(1, 1),
            "must have the same batch",
        ),
    ],
)
def test_recurrent_token_mask_rejects_invalid_shapes(
    positions,
    state_write_valid,
    message,
):
    with pytest.raises(ValueError, match=message):
        model_bf16.build_recurrent_token_mask(positions, state_write_valid)


def test_bind_kv_cache_keeps_runner_pages_and_shared_pool_identity():
    config = _config([LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION])
    specs = model_bf16.build_qwen3_5_kv_spec(config, tp_size=1).layers
    raw, shared_states = _state_views(specs[0])
    key = torch.zeros(4, 1, 2, config.head_dim)
    value = torch.zeros_like(key)
    kv_caches = {
        specs[0].name: shared_states,
        specs[1].name: shared_states,
        specs[2].name: [key, value],
    }
    layers = [
        SimpleNamespace(
            linear_attn=SimpleNamespace(
                state_cache_bases=(),
                state_cache_layout=(),
                state_cache_group=None,
            )
        ),
        SimpleNamespace(
            linear_attn=SimpleNamespace(
                state_cache_bases=(),
                state_cache_layout=(),
                state_cache_group=None,
            )
        ),
        SimpleNamespace(self_attn=SimpleNamespace(k_cache=None, v_cache=None)),
    ]
    model = model_bf16.Qwen3_5ForCausalLM.__new__(model_bf16.Qwen3_5ForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.language_model = SimpleNamespace(layers=layers)

    model.bind_kv_cache(kv_caches)

    first = layers[0].linear_attn
    second = layers[1].linear_attn
    assert first.state_cache_group == second.state_cache_group
    assert first.state_cache_bases[0].untyped_storage().data_ptr() == (
        raw.untyped_storage().data_ptr()
    )
    assert second.state_cache_bases[0] is first.state_cache_bases[0]
    assert layers[2].self_attn.k_cache is key
    assert layers[2].self_attn.v_cache is value


def test_full_attention_prefill_and_decode_use_scheduler_pages():
    torch.manual_seed(42)
    config = _config([FULL_ATTENTION])
    paged = model_bf16.Qwen3_5Attention(config, layer_idx=0)
    reference = model_bf16.Qwen3_5Attention(config, layer_idx=0)
    for parameter in paged.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)
    reference.load_state_dict(paged.state_dict())

    block_size = 2
    paged.k_cache = torch.zeros(4, 1, block_size, config.head_dim)
    paged.v_cache = torch.zeros_like(paged.k_cache)
    paged.k_cache[3].fill_(17)
    paged.v_cache[3].fill_(19)
    untouched_key = paged.k_cache[3].clone()
    untouched_value = paged.v_cache[3].clone()

    first_hidden = torch.randn(1, 3, config.hidden_size)
    first_positions = torch.tensor([[0, 1, 2]])
    rotary = model_bf16.Qwen3_5RotaryEmbedding(config)
    first_rope = rotary(first_positions, first_hidden.dtype)
    first_mask = model_bf16.build_causal_mask(
        3,
        3,
        first_hidden.dtype,
        first_hidden,
    )
    expected_first = reference(first_hidden, first_rope, first_mask)
    actual_first = paged(
        first_hidden,
        first_rope,
        first_mask,
        {
            "slot_mapping": torch.tensor([2, 3, 4]),
            "block_size": block_size,
            "block_table_tensor": torch.tensor([[1, 2]], dtype=torch.int32),
        },
        first_positions,
    )

    second_hidden = torch.randn(1, 1, config.hidden_size)
    second_positions = torch.tensor([[3]])
    second_rope = rotary(second_positions, second_hidden.dtype)
    second_mask = model_bf16.build_causal_mask(
        1,
        1,
        second_hidden.dtype,
        second_hidden,
    )
    full_hidden = torch.cat((first_hidden, second_hidden), dim=1)
    full_positions = torch.arange(4).reshape(1, 4)
    expected_second = reference(
        full_hidden,
        rotary(full_positions, full_hidden.dtype),
        model_bf16.build_causal_mask(
            4,
            4,
            full_hidden.dtype,
            full_hidden,
        ),
    )[:, -1:]
    actual_second = paged(
        second_hidden,
        second_rope,
        second_mask,
        {
            "slot_mapping": torch.tensor([5]),
            "block_size": block_size,
            "block_table_tensor": torch.tensor([[1, 2]], dtype=torch.int32),
        },
        second_positions,
    )

    torch.testing.assert_close(actual_first, expected_first)
    torch.testing.assert_close(actual_second, expected_second)
    torch.testing.assert_close(paged.k_cache[3], untouched_key)
    torch.testing.assert_close(paged.v_cache[3], untouched_value)


class _FakeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(128, dtype=torch.float32).view(16, 8))

    def forward(self, token_ids, **kwargs):
        return self.weight[token_ids]


class _FakeRotary(nn.Module):
    def forward(self, position_ids, dtype):
        shape = (*position_ids.shape, 1)
        return torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)


class _Zero(nn.Module):
    def forward(self, hidden_states, *args):
        return torch.zeros_like(hidden_states)


class _OrderedRecurrentMixer(nn.Module):
    def __init__(self, events: list[str], layer_idx: int):
        super().__init__()
        self.events = events
        self.layer_idx = layer_idx
        self.state_cache_bases = (torch.zeros(8),)
        self.state_cache_layout = ()
        self.state_cache_group = layer_idx

    def paged_state_read_request(self, metadata):
        self.events.append(f"read-{self.layer_idx}")
        return (), metadata["recurrent_state_read_page_ids"], None

    def unpack_paged_state(self, pages):
        return pages[0], pages[1]

    def paged_state_update(self, metadata, state):
        self.events.append(f"write-{self.layer_idx}")
        return (), metadata["recurrent_state_write_page_ids"], state

    def forward(
        self,
        hidden_states,
        *,
        initial_state=None,
        return_state=False,
        **kwargs,
    ):
        self.events.append(f"compute-{self.layer_idx}")
        state = (
            torch.zeros(hidden_states.shape[0], 1),
            torch.zeros(hidden_states.shape[0], 1),
        )
        return (hidden_states, state) if return_state else hidden_states


class _OrderedAttention(nn.Module):
    def __init__(self, events: list[str], layer_idx: int):
        super().__init__()
        self.events = events
        self.layer_idx = layer_idx

    def forward(self, hidden_states, *args, **kwargs):
        self.events.append(f"compute-{self.layer_idx}")
        return torch.zeros_like(hidden_states)


def _ordered_layer(mixer, layer_type: str):
    layer = model_bf16.Qwen3_5DecoderLayer.__new__(model_bf16.Qwen3_5DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_type = layer_type
    layer.input_layernorm = nn.Identity()
    if layer_type == LINEAR_ATTENTION:
        layer.linear_attn = mixer
    else:
        layer.self_attn = mixer
    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = _Zero()
    return layer


def test_model_reads_all_recurrent_states_before_compute_and_writes_after(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        model_bf16,
        "read_mamba_state_pages_grouped",
        lambda bases, reads: [
            [torch.zeros(2, 1), torch.zeros(2, 1)] for _read in reads
        ],
    )
    monkeypatch.setattr(
        model_bf16,
        "write_mamba_state_pages_grouped",
        lambda bases, updates: None,
    )
    layers = [
        _ordered_layer(_OrderedRecurrentMixer(events, 0), LINEAR_ATTENTION),
        _ordered_layer(_OrderedAttention(events, 1), FULL_ATTENTION),
        _ordered_layer(_OrderedRecurrentMixer(events, 2), LINEAR_ATTENTION),
    ]
    model = model_bf16.Qwen3_5Model.__new__(model_bf16.Qwen3_5Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        layer_types=[LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION]
    )
    model.embed_tokens = _FakeEmbedding()
    model.rotary_emb = _FakeRotary()
    model.layers = nn.ModuleList(layers)
    model.norm = nn.Identity()
    metadata = {
        "layers.0.linear_attn": {
            "block_table_tensor": torch.tensor([[1], [2]]),
            "recurrent_state_read_page_ids": torch.tensor([[1], [2]]),
            "recurrent_state_write_page_ids": torch.tensor([[1], [2]]),
            "recurrent_state_write_valid_mask": torch.tensor([[True], [True]]),
        },
        "layers.1.self_attn": {
            "block_table_tensor": torch.tensor([[1], [2]]),
        },
        "layers.2.linear_attn": {
            "block_table_tensor": torch.tensor([[1], [2]]),
            "recurrent_state_read_page_ids": torch.tensor([[1], [2]]),
            "recurrent_state_write_page_ids": torch.tensor([[1], [2]]),
            "recurrent_state_write_valid_mask": torch.tensor([[True], [True]]),
        },
    }

    model(
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        metadata,
        torch.zeros(3, 2, dtype=torch.long),
    )

    assert events == [
        "read-0",
        "read-2",
        "compute-0",
        "compute-1",
        "compute-2",
        "write-0",
        "write-2",
    ]


def test_gdn_prefill_then_decode_updates_only_the_selected_state_page():
    torch.manual_seed(43)
    config = _config([LINEAR_ATTENTION])
    model = model_bf16.Qwen3_5Model(config)
    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)

    layer_spec = model_bf16.build_qwen3_5_kv_spec(config, tp_size=1).layers[0]
    raw, views = _state_views(layer_spec)
    bases, layout = model_bf16.prepare_mamba_state_cache_access(views)
    mixer = model.layers[0].linear_attn
    mixer.state_cache_bases = tuple(bases)
    mixer.state_cache_layout = layout
    mixer.state_cache_group = 0
    page_size = raw.numel() // 4
    page_one = raw[page_size : 2 * page_size].clone()

    prefill_metadata = {
        "layers.0.linear_attn": {
            "block_table_tensor": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_read_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_write_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_cold": torch.tensor([True]),
            "recurrent_state_write_valid_mask": torch.tensor([[True]]),
        }
    }
    decode_metadata = {
        "layers.0.linear_attn": {
            "block_table_tensor": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_read_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_write_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_cold": torch.tensor([False]),
            "recurrent_state_write_valid_mask": torch.tensor([[True]]),
        }
    }

    prefill = model(
        torch.tensor([1, 2, 3]),
        torch.tensor([0, 1, 2]),
        prefill_metadata,
        torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
    )
    page_two_after_prefill = raw[2 * page_size : 3 * page_size].clone()
    decode = model(
        torch.tensor([4]),
        torch.tensor([3]),
        decode_metadata,
        torch.tensor([[3], [3], [3]]),
    )

    assert prefill.shape == (1, 3, config.hidden_size)
    assert decode.shape == (1, 1, config.hidden_size)
    assert torch.isfinite(prefill).all()
    assert torch.isfinite(decode).all()
    assert not torch.equal(
        raw[2 * page_size : 3 * page_size],
        page_two_after_prefill,
    )
    torch.testing.assert_close(raw[page_size : 2 * page_size], page_one)


def test_gdn_packs_checkpoint_stacked_states_for_each_write_page():
    """A [B, S, C, W] checkpoint state writes one page for each checkpoint."""
    torch.manual_seed(47)
    config = _config([LINEAR_ATTENTION])
    model = model_bf16.Qwen3_5Model(config)
    layer_spec = model_bf16.build_qwen3_5_kv_spec(config, tp_size=1).layers[0]
    _raw, views = _state_views(layer_spec)
    bases, layout = model_bf16.prepare_mamba_state_cache_access(views)
    mixer = model.layers[0].linear_attn
    mixer.state_cache_bases = tuple(bases)
    mixer.state_cache_layout = layout
    conv_page_shape, recurrent_page_shape = layer_spec.state_shapes
    # The module keeps the last two axes of each page in the other order.
    conv = torch.randn(1, 2, *reversed(conv_page_shape))
    recurrent = torch.randn(
        1, 2, *recurrent_page_shape[:-2], *reversed(recurrent_page_shape[-2:])
    )
    metadata = {
        "recurrent_state_write_page_ids": torch.tensor([[2, 3]], dtype=torch.int32)
    }

    update = mixer.paged_state_update(metadata, (conv, recurrent))
    model_bf16.write_mamba_state_pages_grouped(mixer.state_cache_bases, [update])

    for checkpoint, page in enumerate((2, 3)):
        torch.testing.assert_close(
            views[0][page], conv[0, checkpoint].transpose(-1, -2)
        )
        torch.testing.assert_close(
            views[1][page], recurrent[0, checkpoint].transpose(-1, -2)
        )


def test_hybrid_model_runs_prefill_and_decode_with_paged_cache_metadata():
    torch.manual_seed(44)
    config = _config([LINEAR_ATTENTION, FULL_ATTENTION])
    model = model_bf16.Qwen3_5ForCausalLM(config, neuron_config=None)
    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)

    specs = model.get_kv_spec().layers
    _raw, recurrent_views = _state_views(specs[0])
    block_size = 2
    key = torch.zeros(4, 1, block_size, config.head_dim)
    value = torch.zeros_like(key)
    model.bind_kv_cache(
        {
            specs[0].name: recurrent_views,
            specs[1].name: [key, value],
        }
    )
    model = torch.compile(model, backend="aot_eager", fullgraph=True)
    prefill_metadata = {
        specs[0].name: {
            "block_table_tensor": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_read_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_write_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_cold": torch.tensor([True]),
            "recurrent_state_write_valid_mask": torch.tensor([[True]]),
        },
        specs[1].name: {
            "block_table_tensor": torch.tensor([[1, 2]], dtype=torch.int32),
            "slot_mapping": torch.tensor([2, 3, 4]),
            "block_size": block_size,
        },
    }
    decode_metadata = {
        specs[0].name: {
            "block_table_tensor": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_read_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_write_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_cold": torch.tensor([False]),
            "recurrent_state_write_valid_mask": torch.tensor([[True]]),
        },
        specs[1].name: {
            "block_table_tensor": torch.tensor([[1, 2]], dtype=torch.int32),
            "slot_mapping": torch.tensor([5]),
            "block_size": block_size,
        },
    }

    prefill = model(
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.tensor([0, 1, 2]),
        rotary_position_ids=torch.tensor(
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        ),
        attn_metadata=prefill_metadata,
        sampling_positions=torch.tensor([2]),
    )
    decode = model(
        input_ids=torch.tensor([4]),
        positions=torch.tensor([3]),
        rotary_position_ids=torch.tensor([[3], [3], [3]]),
        attn_metadata=decode_metadata,
        sampling_positions=torch.tensor([0]),
    )

    assert prefill.shape == (1, config.vocab_size)
    assert decode.shape == (1, config.vocab_size)
    assert torch.isfinite(prefill).all()
    assert torch.isfinite(decode).all()
