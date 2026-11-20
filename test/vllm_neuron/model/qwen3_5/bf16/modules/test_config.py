# SPDX-License-Identifier: Apache-2.0
"""Config tests for the Qwen3.8 hybrid model.

Validates that Qwen3_5TextConfig.from_configs reads the HF config, and that the
layer pattern and the derived sizes match the checkpoint.

The checkpoint config is multimodal. It holds the text fields in a nested
``text_config``, and it holds the rotary fields in a nested ``rope_parameters``.
These tests check that the port reads both nested blocks.

The tests read no checkpoint and need no device, so ``make test`` runs them.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        test/vllm_neuron/model/qwen3_5/bf16/modules/test_config.py -v
"""

import pytest
import torch

from vllm_neuron.model.qwen3_5.config import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen3_5TextConfig,
)

pytestmark = [pytest.mark.fast]

# The text block of the Qwen/Qwen3.8-27B config, cut to the fields that matter.
QWEN3_8_27B_TEXT = {
    "vocab_size": 248320,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 262144,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attn_output_gate": True,
    "full_attention_interval": 4,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "output_gate_type": "swish",
    "mamba_ssm_dtype": "float32",
    "dtype": "bfloat16",
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
}


class _FakeHfConfig:
    """Minimal stand-in for a checkpoint config object."""

    def __init__(self, data: dict, nested: bool):
        self._data = dict(data)
        if nested:
            self.text_config = _FakeHfConfig(data, nested=False)

    def to_dict(self) -> dict:
        return dict(self._data)


def test_reads_the_nested_text_config():
    config = Qwen3_5TextConfig.from_configs(
        _FakeHfConfig(QWEN3_8_27B_TEXT, nested=True)
    )

    assert config.num_hidden_layers == 64
    assert config.hidden_size == 5120
    assert config.num_attention_heads == 24
    assert config.num_key_value_heads == 4
    assert config.head_dim == 256
    assert config.linear_num_key_heads == 16
    assert config.linear_num_value_heads == 48
    assert config.torch_dtype is torch.bfloat16
    assert config.mamba_ssm_dtype is torch.float32


def test_reads_the_nested_rope_parameters():
    config = Qwen3_5TextConfig.from_configs(
        _FakeHfConfig(QWEN3_8_27B_TEXT, nested=True)
    )

    assert config.rope_theta == 1.0e7
    assert config.partial_rotary_factor == 0.25
    assert config.mrope_section == [11, 11, 10]
    # The model rotates one quarter of each head.
    assert config.rotary_dim == 64
    # The three mRoPE sections cover the frequency count of the rotated part.
    assert sum(config.mrope_section) == config.rotary_dim // 2


def test_accepts_a_text_only_config():
    """A text-only run passes the text config directly, with no nesting."""
    config = Qwen3_5TextConfig.from_configs(
        _FakeHfConfig(QWEN3_8_27B_TEXT, nested=False)
    )
    assert config.num_hidden_layers == 64
    assert config.rope_theta == 1.0e7


def test_derived_sizes_match_the_checkpoint():
    config = Qwen3_5TextConfig.from_configs(
        _FakeHfConfig(QWEN3_8_27B_TEXT, nested=True)
    )

    assert config.key_dim == 2048  # 16 key heads x 128
    assert config.value_dim == 6144  # 48 value heads x 128
    # The convolution covers q, k and v. The checkpoint tensor width is 10240.
    assert config.conv_dim == 10240


def test_layer_pattern_puts_one_full_attention_layer_in_each_group_of_four():
    config = Qwen3_5TextConfig.from_configs(
        _FakeHfConfig(QWEN3_8_27B_TEXT, nested=True)
    )

    assert config.layer_types[:4] == [
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        FULL_ATTENTION,
    ]
    assert config.num_linear_layers == 48
    assert config.num_full_attention_layers == 16


def test_explicit_layer_types_win_over_the_interval():
    data = dict(QWEN3_8_27B_TEXT)
    data["num_hidden_layers"] = 4
    data["layer_types"] = [
        FULL_ATTENTION,
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
    ]
    config = Qwen3_5TextConfig.from_configs(_FakeHfConfig(data, nested=True))
    assert config.layer_types[0] == FULL_ATTENTION
    assert config.num_full_attention_layers == 1
