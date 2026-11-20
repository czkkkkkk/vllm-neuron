# SPDX-License-Identifier: Apache-2.0
"""Weight-map tests for the Qwen3.8 hybrid model.

Validates build_weight_mappings and UNUSED_HF_PREFIXES against the checkpoint
tensor names.

The loader ends with ``load_state_dict(strict=False)``. A tensor that no entry of
the map covers is then dropped in silence, and a parameter keeps its empty
buffer. These tests close that gap. They check the map against the tensor names
of the checkpoint.

The checkpoint index test needs a local checkpoint. Point
``QWEN3_8_CHECKPOINT_DIR`` at the directory that holds
``model.safetensors.index.json``. The test skips when the variable is not set.

The default tests read no checkpoint and need no device, so ``make test`` runs
them.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        test/vllm_neuron/model/qwen3_5/bf16/modules/test_weight_mappings.py -v
"""

import json
import os
import pathlib

import pytest

from vllm_neuron.model.qwen3_5.config import FULL_ATTENTION, Qwen3_5TextConfig
from vllm_neuron.model.qwen3_5.model_bf16 import (
    HF_TEXT_PREFIX,
    UNUSED_HF_PREFIXES,
    build_weight_mappings,
)

# Tensor count of Qwen/Qwen3.8-27B. The vision tower holds 333 tensors and the
# multi-token-prediction head holds 15 tensors, so 851 text tensors remain.
QWEN3_8_27B_TOTAL_TENSORS = 1199
QWEN3_8_27B_TEXT_TENSORS = 851
QWEN3_8_27B_UNUSED_TENSORS = 348

pytestmark = [pytest.mark.fast]


@pytest.fixture
def config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig()


def _targets(mappings: dict[str, str | list[str]]) -> list[str]:
    """Return every checkpoint tensor name of the map. A joined weight reads several."""
    names = []
    for target in mappings.values():
        names.extend(target if isinstance(target, list) else [target])
    return names


def test_map_size_matches_the_checkpoint_text_tensor_count(config):
    mappings = build_weight_mappings(config)
    assert len(_targets(mappings)) == QWEN3_8_27B_TEXT_TENSORS
    assert QWEN3_8_27B_TEXT_TENSORS + QWEN3_8_27B_UNUSED_TENSORS == (
        QWEN3_8_27B_TOTAL_TENSORS
    )


def test_every_target_name_is_unique(config):
    """Two parameters must not read the same checkpoint tensor by accident."""
    targets = _targets(build_weight_mappings(config))
    duplicates = {t for t in targets if targets.count(t) > 1}
    assert not duplicates, f"these checkpoint tensors are mapped twice: {duplicates}"


def test_map_holds_no_unused_prefix(config):
    """The map must not reach the vision tower or the draft head."""
    mappings = build_weight_mappings(config)
    for target in mappings.values():
        names = target if isinstance(target, list) else [target]
        for name in names:
            assert not name.startswith(UNUSED_HF_PREFIXES), (
                f"{name} belongs to a tensor group that this port does not load"
            )


def test_each_layer_maps_one_mixer_only(config):
    """A linear layer has no attention tensor, and the reverse also holds."""
    mappings = build_weight_mappings(config)
    for layer_idx, layer_type in enumerate(config.layer_types):
        prefix = f"language_model.layers.{layer_idx}."
        names = [k for k in mappings if k.startswith(prefix)]
        has_attention = any(".self_attn." in k for k in names)
        has_linear = any(".linear_attn." in k for k in names)
        if layer_type == FULL_ATTENTION:
            assert has_attention and not has_linear, f"layer {layer_idx} is wrong"
        else:
            assert has_linear and not has_attention, f"layer {layer_idx} is wrong"


def test_text_tensors_use_the_checkpoint_prefix(config):
    """Every layer tensor sits under the text prefix of the checkpoint."""
    mappings = build_weight_mappings(config)
    for param_name, target in mappings.items():
        if param_name == "lm_head.weight":
            continue
        for name in target if isinstance(target, list) else [target]:
            assert name.startswith(HF_TEXT_PREFIX), (
                f"{param_name} reads {name}, which is outside {HF_TEXT_PREFIX}"
            )


def test_map_covers_the_checkpoint_index_exactly(config):
    """Compare the map against a real checkpoint index.

    The test fails when the map holds a name that the checkpoint does not hold,
    and it fails when the checkpoint holds a text tensor that no entry covers.
    """
    checkpoint_dir = os.environ.get("QWEN3_8_CHECKPOINT_DIR")
    if not checkpoint_dir:
        pytest.skip("Set QWEN3_8_CHECKPOINT_DIR to run the checkpoint index test.")

    index_path = pathlib.Path(checkpoint_dir) / "model.safetensors.index.json"
    if not index_path.is_file():
        pytest.skip(f"{index_path} does not exist.")

    checkpoint_names = set(json.loads(index_path.read_text())["weight_map"])
    mapped = set()
    for target in build_weight_mappings(config).values():
        mapped.update(target if isinstance(target, list) else [target])

    ignored = {n for n in checkpoint_names if n.startswith(UNUSED_HF_PREFIXES)}
    missing = sorted(mapped - checkpoint_names)
    unmapped = sorted(checkpoint_names - mapped - ignored)

    assert not missing, f"the map reads tensors that the checkpoint lacks: {missing}"
    assert not unmapped, (
        f"the checkpoint holds tensors that no entry covers: {unmapped}"
    )


def test_load_weights_rejects_an_fp8_checkpoint_tensor(monkeypatch):
    """The BF16 model must refuse an FP8 tensor instead of casting it.

    An FP8 tensor has no scale in the BF16 model, so a plain cast to BF16 gives
    wrong values with no error.
    """
    from types import SimpleNamespace

    import torch

    from vllm_neuron.model.qwen3_5 import model_bf16

    class _StubGroup:
        world_size, rank_in_group, device_group = 1, 0, None

    class _FakeCheckpoint:
        # Like the real loader, return the tensor after the cast to the
        # parameter dtype. The stored dtype is only in the file header.
        def __init__(self, *args, **kwargs):
            pass

        def load_sharded_pipelined(self, *args, **kwargs):
            cast = torch.zeros(4, dtype=torch.bfloat16)
            return SimpleNamespace(state_dict={"lm_head.weight": cast})

        def tensor_dtypes(self):
            return {"lm_head.weight": "F8_E4M3"}

    monkeypatch.setattr(model_bf16, "get_tp_group", lambda: _StubGroup())
    monkeypatch.setattr(model_bf16, "SafetensorsCheckpoint", _FakeCheckpoint)
    tiny = Qwen3_5TextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        vocab_size=128,
        mrope_section=[3, 3, 2],
    )
    model = model_bf16.Qwen3_5ForCausalLM(tiny, neuron_config=None)
    with pytest.raises(ValueError, match="Use a BF16 checkpoint"):
        model.load_weights("unused", torch.device("cpu"), None)


@pytest.mark.parametrize("tied", [False, True])
def test_lm_head_reads_the_embedding_when_embeddings_are_tied(config, tied):
    from dataclasses import replace

    mappings = build_weight_mappings(replace(config, tie_word_embeddings=tied))

    expected = (
        mappings["language_model.embed_tokens.weight"] if tied else "lm_head.weight"
    )
    assert mappings["lm_head.weight"] == expected
