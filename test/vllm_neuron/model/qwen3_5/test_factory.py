# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.8 BF16 factory and registry."""

from types import SimpleNamespace

import pytest
from vllm.v1.kv_cache_interface import MambaAttentionBackendEnum

from vllm_neuron.model.qwen3_5.factory import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)

pytestmark = [pytest.mark.fast]


class _FakeHfConfig:
    def __init__(self, data: dict | None = None):
        self._data = {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            **(data or {}),
        }
        self.quantization_config = self._data.get("quantization_config")

    def to_dict(self) -> dict:
        return dict(self._data)


class _FakeNeuronConfig:
    def __init__(self, quantization):
        self.quantization = quantization


@pytest.mark.parametrize("quantization", [None, "bf16"])
def test_factory_accepts_bf16_quantization(quantization):
    Qwen3_5ForConditionalGeneration._validate_config(
        _FakeHfConfig(),
        _FakeNeuronConfig(quantization),
    )


def test_factory_rejects_a_quantized_checkpoint():
    hf_config = _FakeHfConfig({"quantization_config": {"quant_method": "fp8"}})

    with pytest.raises(ValueError, match="checkpoint is quantized"):
        Qwen3_5ForConditionalGeneration._validate_config(
            hf_config,
            _FakeNeuronConfig(None),
        )


def test_factory_rejects_an_unsupported_quantization():
    with pytest.raises(ValueError, match="does not support quantization"):
        Qwen3_5ForConditionalGeneration._validate_config(
            _FakeHfConfig(),
            _FakeNeuronConfig("mxfp8"),
        )


def test_factory_rejects_a_config_without_hybrid_fields():
    hf_config = _FakeHfConfig()
    del hf_config._data["linear_num_value_heads"]

    with pytest.raises(ValueError, match="linear_num_value_heads"):
        Qwen3_5ForConditionalGeneration._validate_config(
            hf_config,
            _FakeNeuronConfig(None),
        )


def test_factory_accepts_the_multimodal_runner_arguments(monkeypatch):
    selected_model = object()
    calls = []

    def select(cls, hf_config, text_neuron_config, encoder_neuron_config):
        calls.append((hf_config, text_neuron_config, encoder_neuron_config))
        return selected_model

    monkeypatch.setattr(
        Qwen3_5ForConditionalGeneration,
        "_select_implementation",
        classmethod(select),
    )
    inputs = (object(), object(), object())

    actual = Qwen3_5ForConditionalGeneration.from_configs(
        hf_config=inputs[0],
        text_neuron_config=inputs[1],
        encoder_neuron_config=inputs[2],
    )

    assert actual is selected_model
    assert calls == [inputs]


def test_factory_reports_per_rank_gdn_cache_geometry():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_FakeHfConfig()),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
    )

    specs = Qwen3_5ForConditionalGeneration.get_mamba_specs_from_config(vllm_config)

    assert len(specs) == 48
    assert specs[0].shapes == ((3, 1280), (6, 128, 128))
    assert specs[0].page_size_bytes == 400896
    assert specs[0].mamba_type is MambaAttentionBackendEnum.GDN_ATTN
    assert specs[0].tp_replicated is False


@pytest.mark.parametrize(
    "factory",
    [Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM],
)
def test_registry_routes_qwen3_5_to_the_factory(factory):
    from vllm_neuron.model.registry import get_models

    # The registry key and the class name are the same, as in vLLM.
    assert dict(get_models())[factory.__name__] is factory
    assert issubclass(factory, Qwen3_5ForConditionalGeneration)


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "fp8"])
def test_runner_policy_restricts_qwen_kv_cache_to_bf16(cache_dtype):
    from vllm_neuron.vllm.model_runner_v2_policy import (
        get_neuron_model_runner_v2_policy,
    )

    def unsupported(arch):
        config = SimpleNamespace(
            model_config=SimpleNamespace(architectures=[arch], runner_type="generate"),
            cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        )
        return get_neuron_model_runner_v2_policy(config).unsupported_features

    reason = "non-BF16 KV cache on Qwen3.8"
    assert (reason in unsupported("Qwen3_5ForCausalLM")) == (cache_dtype == "fp8")
    assert reason not in unsupported("LlamaForCausalLM")
