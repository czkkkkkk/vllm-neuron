# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.8 BF16 core and construction hooks."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_neuron.model.qwen3_5.model_bf16 as model_bf16
from vllm_neuron.model.qwen3_5.config import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen3_5TextConfig,
)

pytestmark = [pytest.mark.fast]


def _config(layer_types: list[str] | None = None) -> Qwen3_5TextConfig:
    if layer_types is None:
        layer_types = [LINEAR_ATTENTION]
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


def test_attention_calls_the_projection_weight_hook():
    class _Attention(model_bf16.Qwen3_5Attention):
        def _create_projection_weights(self, q_out, kv_out, o_in):
            self.projection_shape = (q_out, kv_out, o_in)
            super()._create_projection_weights(q_out, kv_out, o_in)

    attention = _Attention(_config([FULL_ATTENTION]), layer_idx=0)

    assert attention.projection_shape == (32, 8, 16)


def test_gdn_calls_the_projection_weight_hook():
    class _GDN(model_bf16.Qwen3_5GatedDeltaNet):
        def _create_projection_weights(self):
            self.created_projection_weights = True
            super()._create_projection_weights()

    mixer = _GDN(_config(), layer_idx=0)

    assert mixer.created_projection_weights


def test_decoder_layer_uses_the_selected_mixer_and_mlp_classes():
    class _Attention(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx

    class _GDN(_Attention):
        pass

    class _MLP(nn.Module):
        def __init__(self, config):
            super().__init__()

    class _Layer(model_bf16.Qwen3_5DecoderLayer):
        attention_cls = _Attention
        gated_delta_net_cls = _GDN
        mlp_cls = _MLP

    linear = _Layer(_config([LINEAR_ATTENTION]), layer_idx=0)
    attention = _Layer(_config([FULL_ATTENTION]), layer_idx=0)

    assert isinstance(linear.linear_attn, _GDN)
    assert isinstance(linear.mlp, _MLP)
    assert isinstance(attention.self_attn, _Attention)


def test_backbone_uses_the_selected_decoder_layer_class(monkeypatch):
    class _Embedding(nn.Module):
        def __init__(self, vocab_size, embed_dim, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(vocab_size, embed_dim))

        def forward(self, token_ids, **kwargs):
            return self.weight[token_ids]

    class _Layer(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx

    class _Model(model_bf16.Qwen3_5Model):
        decoder_layer_cls = _Layer

    monkeypatch.setattr(model_bf16, "VocabDimShardedEmbedding", _Embedding)
    model = _Model(_config([LINEAR_ATTENTION, FULL_ATTENTION]))

    assert [layer.layer_idx for layer in model.layers] == [0, 1]


def test_causal_lm_uses_the_selected_backbone_class(monkeypatch):
    class _Backbone(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    class _Linear(nn.Linear):
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__(
                in_features,
                out_features,
                bias=kwargs.get("bias", False),
                dtype=kwargs.get("dtype"),
            )

    class _CausalLM(model_bf16.Qwen3_5ForCausalLM):
        backbone_cls = _Backbone

    monkeypatch.setattr(model_bf16.neuron_nn, "ColumnParallelLinear", _Linear)
    model = _CausalLM(_config(), neuron_config=None)

    assert isinstance(model.language_model, _Backbone)


def test_gdn_state_continues_across_calls():
    torch.manual_seed(42)
    mixer = model_bf16.Qwen3_5GatedDeltaNet(_config(), layer_idx=0)
    for parameter in mixer.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)
    hidden = torch.randn(2, 5, 16)

    whole, whole_state = mixer(hidden, return_state=True)
    first, first_state = mixer(hidden[:, :4], return_state=True)
    second, second_state = mixer(
        hidden[:, 4:],
        initial_state=first_state,
        return_state=True,
    )

    torch.testing.assert_close(
        torch.cat((first, second), dim=1),
        whole,
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(second_state[0], whole_state[0])
    torch.testing.assert_close(second_state[1], whole_state[1], rtol=2e-4, atol=2e-4)


def test_gdn_returns_one_state_checkpoint_for_each_token():
    torch.manual_seed(42)
    mixer = model_bf16.Qwen3_5GatedDeltaNet(_config(), layer_idx=0)
    hidden = torch.randn(2, 3, 16)

    output, (conv_states, recurrent_states) = mixer(
        hidden,
        return_state=True,
        return_state_checkpoints=True,
    )

    assert output.shape == (2, 3, 16)
    assert conv_states.shape == (2, 3, 32, 3)
    assert recurrent_states.shape == (2, 3, 4, 4, 4)


def test_gdn_rejects_checkpoint_state_without_a_state_return():
    mixer = model_bf16.Qwen3_5GatedDeltaNet(_config(), layer_idx=0)

    with pytest.raises(ValueError, match="Set return_state=True"):
        mixer(
            torch.randn(1, 2, 16),
            return_state_checkpoints=True,
        )


def test_zero_centered_rms_norm_matches_the_reference_formula():
    norm = model_bf16.Qwen3_5RMSNorm(4, eps=1e-6, dtype=torch.float32)
    norm.weight.data.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    hidden = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

    actual = norm(hidden)

    expected = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected * (1.0 + norm.weight)
    torch.testing.assert_close(actual, expected)


def test_gated_rms_norm_applies_silu_after_normalization():
    norm = model_bf16.Qwen3_5RMSNormGated(4, eps=1e-6, dtype=torch.float32)
    norm.weight.data.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))
    hidden = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    gate = torch.tensor([[0.2, -0.3, 0.4, -0.5]])

    actual = norm(hidden, gate)

    expected = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected * norm.weight * torch.nn.functional.silu(gate)
    torch.testing.assert_close(actual, expected)


def test_partial_rotary_embedding_preserves_unrotated_channels():
    query = torch.randn(2, 3, 2, 8)
    key = torch.randn(2, 3, 1, 8)
    cos = torch.randn(2, 3, 2)
    sin = torch.randn(2, 3, 2)

    rotated_query, rotated_key = model_bf16.apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
    )

    torch.testing.assert_close(rotated_query[..., 2:], query[..., 2:])
    torch.testing.assert_close(rotated_key[..., 2:], key[..., 2:])


def test_attention_is_causal_and_keeps_the_expected_shape():
    torch.manual_seed(42)
    attention = model_bf16.Qwen3_5Attention(
        _config([FULL_ATTENTION]),
        layer_idx=0,
    )
    for parameter in attention.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)
    hidden = torch.randn(1, 3, 16)
    changed = hidden.clone()
    changed[:, 1:] += 100.0
    rotary = model_bf16.Qwen3_5RotaryEmbedding(_config([FULL_ATTENTION]))
    positions = torch.arange(3).view(1, -1)
    position_embeddings = rotary(positions, hidden.dtype)
    mask = model_bf16.build_causal_mask(3, 3, hidden.dtype, hidden)

    output = attention(hidden, position_embeddings, mask)
    changed_output = attention(changed, position_embeddings, mask)

    assert output.shape == hidden.shape
    torch.testing.assert_close(output[:, :1], changed_output[:, :1])


def test_mlp_torch_path_matches_the_projection_formula():
    torch.manual_seed(42)
    mlp = model_bf16.Qwen3_5MLP(_config())
    for parameter in mlp.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)
    hidden = torch.randn(2, 3, 16)

    actual = mlp(hidden)

    expected = torch.nn.functional.silu(hidden @ mlp.gate_proj_weight)
    expected = expected * (hidden @ mlp.up_proj_weight)
    expected = expected @ mlp.down_proj_weight
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("tp_rank,expected_token", [(0, 3), (1, 0)])
def test_vocab_restrictions_before_sampling(tp_rank, expected_token):
    """Masked IDs are global; the compiled model takes TP rank as an input."""
    class Backbone(nn.Module):
        def forward(self, input_ids, *args, **kwargs):
            return torch.arange(4, dtype=torch.float32).expand(*input_ids.shape, 4)

    class Greedy(nn.Module):
        def forward(self, logits, *args, **kwargs):
            return logits.argmax(dim=-1)

    model = model_bf16.Qwen3_5ForCausalLM.__new__(model_bf16.Qwen3_5ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=8)
    model.sampling_vocab_size = 6
    model.suppressed_token_ids = (5,)
    model.return_mtp_hidden_states = False
    model.on_device_sampling_config = {}
    model._gather_logits = False
    model.lm_head = nn.Identity()
    model.language_model = Backbone()
    model.sampler = Greedy()
    ids = torch.tensor([[1, 2]])
    rank = torch.tensor([tp_rank], dtype=torch.int32)
    sampled, _ = model(ids, sampling_positions=torch.tensor([1]), rank=rank)
    assert sampled.tolist() == [expected_token]
    logits = model(ids, rank=rank)
    assert logits.shape == (1, 2, 4)
    assert logits.argmax(dim=-1).tolist() == [[expected_token] * 2]
