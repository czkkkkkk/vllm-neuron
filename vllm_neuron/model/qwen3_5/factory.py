# SPDX-License-Identifier: Apache-2.0
"""Factory for the Qwen3.5 architecture used by Qwen3.8 checkpoints."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig

from .config import Qwen3_5TextConfig

# Quantization values that this model accepts. Only bfloat16 exists.
_SUPPORTED_QUANTIZATION = (None, "bf16")


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Factory that validates config and selects the Qwen3.5 implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements. The registry
    uses the same Qwen3_5ForConditionalGeneration class name as upstream vLLM.
    Qwen3.8 checkpoints declare this architecture. The checkpoint is multimodal,
    but the model runs text only.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config,
            text_neuron_config,
            encoder_neuron_config,
        )

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> nn.Module:
        """Create the BF16 text model from the runner configuration."""
        return cls._select_implementation(
            hf_config,
            text_neuron_config,
            encoder_neuron_config,
        )

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config):
        """Return per-rank Gated DeltaNet state geometry for cache alignment."""
        from vllm.v1.kv_cache_interface import MambaSpec

        from vllm_neuron.model.kv_cache import RecurrentLayerSpec

        from .model_bf16 import build_qwen3_5_kv_spec

        config = Qwen3_5TextConfig.from_configs(vllm_config.model_config.hf_config)
        kv_spec = build_qwen3_5_kv_spec(
            config,
            vllm_config.parallel_config.tensor_parallel_size,
        )
        return [
            MambaSpec(
                block_size=-1,
                shapes=layer.state_shapes,
                dtypes=layer.state_dtypes,
                mamba_type=layer.state_backend_type,
                num_prefill_checkpoint_blocks=(
                    layer.state_num_prefill_checkpoint_blocks
                ),
                num_heads=layer.state_num_heads,
                tokens_per_state=layer.state_tokens_per_state,
                tp_replicated=layer.state_tp_replicated,
            )
            for layer in kv_spec.layers
            if isinstance(layer, RecurrentLayerSpec)
        ]

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        encoder_neuron_config: EncoderNeuronConfig | None,
    ) -> nn.Module:
        """Validate the configuration and build the BF16 text model."""
        cls._validate_config(hf_config, text_neuron_config)

        from .model_bf16 import Qwen3_5ForCausalLM as Model

        # The checkpoint includes a vision encoder. The model runs text only.
        del encoder_neuron_config
        config = Qwen3_5TextConfig.from_configs(hf_config, text_neuron_config)
        return Model.from_configs(config, text_neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Reject quantized checkpoints and configs without hybrid fields."""
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in _SUPPORTED_QUANTIZATION:
            raise ValueError(
                f"Qwen3.8 BF16 does not support quantization={quantization!r}. "
                "Use None or 'bf16'."
            )

        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is None and hasattr(hf_config, "to_dict"):
            quant_cfg = hf_config.to_dict().get("quantization_config")
        if quant_cfg is not None:
            raise ValueError(
                "The Qwen3.8 checkpoint is quantized. Use a BF16 checkpoint."
            )

        # The config must hold the Gated DeltaNet fields. Without them,
        # Qwen3_5TextConfig uses the 27B defaults and loads the wrong shapes.
        text = getattr(hf_config, "text_config", None) or hf_config
        raw = text.to_dict() if hasattr(text, "to_dict") else dict(text)
        for required in ("linear_num_key_heads", "linear_num_value_heads"):
            if raw.get(required) is None:
                raise ValueError(
                    f"The checkpoint config has no {required!r} field. "
                    "Check the --model argument."
                )


class Qwen3_5ForCausalLM(Qwen3_5ForConditionalGeneration):
    """Factory for a checkpoint that declares ``Qwen3_5ForCausalLM``.

    vLLM uses this architecture name for the Qwen3.5 text model. The factory
    selects the same text implementation as ``Qwen3_5ForConditionalGeneration``.
    The model loads the multimodal checkpoint layout, with the text weights
    under ``model.language_model``.
    """
