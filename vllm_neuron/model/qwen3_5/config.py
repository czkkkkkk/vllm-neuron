# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8 Config
==============

<-- MODEL-SPECIFIC: All fields in this config are Qwen3.8-specific.
Qwen3.8-27B: 64 layers (48 Gated DeltaNet, 16 full attention), 24 Q heads /
4 KV heads, head_dim=256, hidden=5120.
Key differences from Qwen3:
  - Hybrid layer pattern from layer_types or full_attention_interval
  - Gated DeltaNet uses a 4-token convolution, 16 key heads, 48 value heads,
    128 channels per head, a swish output gate, and an FP32 recurrent state.
  - Partial mRoPE: partial_rotary_factor=0.25, mrope_section=[11, 11, 10]
  - The HF config nests the text fields in text_config. This config reads that
    block only, because the model runs text only.
"""

from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

# Names that ``layer_types`` uses for the two token mixers.
LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

# dtype names that the HF config can hold for this model.
_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _to_dtype(value: str | torch.dtype | None, default: torch.dtype) -> torch.dtype:
    """Return the torch dtype for an HF config dtype value."""
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if value not in _DTYPE_BY_NAME:
        raise ValueError(
            f"dtype {value!r} is not supported for Qwen3.8. Use one of "
            f"{sorted(_DTYPE_BY_NAME)}."
        )
    return _DTYPE_BY_NAME[value]


@dataclass
class Qwen3_5TextConfig:
    """Text config of the Qwen3.8 hybrid model. The defaults match Qwen3.8-27B."""

    # Backbone.
    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False

    # Optional static token restrictions, applied before sampling and scoring.
    sampling_vocab_size: int | None = None
    suppressed_token_ids: tuple[int, ...] = ()

    # Full-attention layers.
    attention_bias: bool = False
    attn_output_gate: bool = True

    # Gated DeltaNet layers.
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    output_gate_type: str = "swish"

    # Layer pattern. A layer_types list in the checkpoint overrides the interval.
    full_attention_interval: int = 4
    layer_types: list[str] | None = None

    # Partial mRoPE: rotate partial_rotary_factor of each head, split across the
    # three position axes by mrope_section.
    rope_theta: float = 1.0e7
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])

    # dtype of the Gated DeltaNet recurrent state.
    mamba_ssm_dtype: torch.dtype = torch.float32

    # dtype of the weights and activations.
    torch_dtype: torch.dtype = torch.bfloat16

    # Multi-token-prediction head in the checkpoint.
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.layer_types is None:
            # One full-attention layer ends each group of full_attention_interval
            # layers, as in the checkpoint.
            self.layer_types = [
                LINEAR_ATTENTION
                if (i + 1) % self.full_attention_interval
                else FULL_ATTENTION
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types holds {len(self.layer_types)} entries, but the model "
                f"has {self.num_hidden_layers} layers. Fix the checkpoint config."
            )
        unknown = set(self.layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION}
        if unknown:
            raise ValueError(
                f"layer_types holds unknown mixer names {sorted(unknown)}. This model "
                f"supports {LINEAR_ATTENTION!r} and {FULL_ATTENTION!r} only."
            )

    # Derived sizes, for all heads before TP sharding.

    @property
    def key_dim(self) -> int:
        """Total Gated DeltaNet key width."""
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        """Total Gated DeltaNet value width."""
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        """Gated DeltaNet conv1d width: q, k and v together."""
        return 2 * self.key_dim + self.value_dim

    @property
    def rotary_dim(self) -> int:
        """Number of head dims that RoPE rotates."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def num_linear_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == LINEAR_ATTENTION)

    @property
    def num_full_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == FULL_ATTENTION)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: object | None = None
    ) -> "Qwen3_5TextConfig":
        """Build the text config from an HF config.

        hf_config is the multimodal config or its text_config. A field that the
        HF config does not hold takes the Qwen3.8-27B default. neuron_config is
        not used. The factory interface passes it.
        """
        text = getattr(hf_config, "text_config", None) or hf_config
        raw = text.to_dict() if hasattr(text, "to_dict") else dict(text)

        # Newer HF configs hold the rotary fields in rope_parameters, older ones
        # at the top level. Read rope_parameters first.
        rope = raw.get("rope_parameters") or {}

        def pick(key: str, default):
            value = rope.get(key, raw.get(key, default))
            return default if value is None else value

        dtype = _to_dtype(raw.get("torch_dtype") or raw.get("dtype"), torch.bfloat16)
        ssm_dtype = _to_dtype(raw.get("mamba_ssm_dtype"), torch.float32)

        defaults = cls()
        return cls(
            vocab_size=raw.get("vocab_size", defaults.vocab_size),
            sampling_vocab_size=getattr(hf_config, "sampling_vocab_size", None),
            suppressed_token_ids=tuple(getattr(hf_config, "suppressed_token_ids", ())),
            hidden_size=raw.get("hidden_size", defaults.hidden_size),
            intermediate_size=raw.get("intermediate_size", defaults.intermediate_size),
            num_hidden_layers=raw.get("num_hidden_layers", defaults.num_hidden_layers),
            num_attention_heads=raw.get(
                "num_attention_heads", defaults.num_attention_heads
            ),
            num_key_value_heads=raw.get(
                "num_key_value_heads", defaults.num_key_value_heads
            ),
            head_dim=raw.get("head_dim", defaults.head_dim),
            hidden_act=raw.get("hidden_act", defaults.hidden_act),
            rms_norm_eps=raw.get("rms_norm_eps", defaults.rms_norm_eps),
            max_position_embeddings=raw.get(
                "max_position_embeddings", defaults.max_position_embeddings
            ),
            tie_word_embeddings=raw.get(
                "tie_word_embeddings", defaults.tie_word_embeddings
            ),
            attention_bias=raw.get("attention_bias", defaults.attention_bias),
            attn_output_gate=raw.get("attn_output_gate", defaults.attn_output_gate),
            linear_conv_kernel_dim=raw.get(
                "linear_conv_kernel_dim", defaults.linear_conv_kernel_dim
            ),
            linear_key_head_dim=raw.get(
                "linear_key_head_dim", defaults.linear_key_head_dim
            ),
            linear_value_head_dim=raw.get(
                "linear_value_head_dim", defaults.linear_value_head_dim
            ),
            linear_num_key_heads=raw.get(
                "linear_num_key_heads", defaults.linear_num_key_heads
            ),
            linear_num_value_heads=raw.get(
                "linear_num_value_heads", defaults.linear_num_value_heads
            ),
            output_gate_type=raw.get("output_gate_type", defaults.output_gate_type),
            full_attention_interval=raw.get(
                "full_attention_interval", defaults.full_attention_interval
            ),
            layer_types=raw.get("layer_types"),
            rope_theta=float(pick("rope_theta", defaults.rope_theta)),
            partial_rotary_factor=float(
                pick("partial_rotary_factor", defaults.partial_rotary_factor)
            ),
            mrope_section=list(pick("mrope_section", defaults.mrope_section)),
            mamba_ssm_dtype=ssm_dtype,
            torch_dtype=dtype,
            mtp_num_hidden_layers=raw.get(
                "mtp_num_hidden_layers", defaults.mtp_num_hidden_layers
            ),
            mtp_use_dedicated_embeddings=raw.get(
                "mtp_use_dedicated_embeddings",
                defaults.mtp_use_dedicated_embeddings,
            ),
        )
