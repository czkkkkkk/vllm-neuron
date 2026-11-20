# SPDX-License-Identifier: Apache-2.0
"""Compare a tiny Qwen3.8 model with the HF Qwen3.5 text model in FP32.

The other Qwen3.8 tests compare a module with itself or check only shapes. This
test uses an independent reference, so a wrong mRoPE section, norm offset, gate
order, or weight mapping fails.
"""

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")

from safetensors.torch import save_file  # noqa: E402
from transformers.models.qwen3_5 import (  # noqa: E402
    Qwen3_5ForCausalLM as HfQwen3_5ForCausalLM,
)
from transformers.models.qwen3_5 import (  # noqa: E402
    Qwen3_5TextConfig as HfQwen3_5TextConfig,
)

import vllm_neuron.model.qwen3_5.model_bf16 as model_bf16  # noqa: E402
from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint  # noqa: E402
from test.vllm_neuron.model.qwen3_5.bf16.modules.test_cache_binding import (  # noqa: E402
    _single_rank,  # noqa: F401
    _state_views,
)

pytestmark = [pytest.mark.fast]

PREFILL_TOKENS = 5
BLOCK_SIZE = 4
# The text position and the three mRoPE axes (temporal, height, width) of each
# token. The axes differ, so the mRoPE section and the interleave affect the
# output. The height and width positions are large, so each rotary frequency
# turns by a measurable angle.
TEXT_POSITIONS = [0, 1, 2, 3, 4, 5]
MROPE_POSITIONS = [
    [0, 1, 2, 3, 4, 5],
    [0, 0, 8, 8, 16, 16],
    [0, 5, 10, 15, 20, 25],
]
# Both models run in FP32. Their difference is about 5e-7 and comes from a
# different summation order, for example the chunked delta rule. A wrong mRoPE
# section, norm offset, or gate order changes the logits by more than 1e-4.
ATOL = 1e-5
RTOL = 1e-5


def _hf_config(tied: bool = False) -> HfQwen3_5TextConfig:
    return HfQwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        rope_parameters={
            "rope_type": "default",
            # A small theta makes the slow frequencies turn faster.
            "rope_theta": 100.0,
            "partial_rotary_factor": 0.5,
            # 8 rotary frequencies. Each axis owns more than one frequency, so
            # a wrong section moves a frequency to another axis.
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        },
        tie_word_embeddings=tied,
        dtype="float32",
    )


def _hf_model(config: HfQwen3_5TextConfig) -> HfQwen3_5ForCausalLM:
    torch.manual_seed(45)
    model = HfQwen3_5ForCausalLM(config).to(torch.float32).eval()
    # The HF init sets each norm weight to zero. Random values make the norm
    # offset and the gates affect the output.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    return model


def _save_multimodal_layout(
    model: HfQwen3_5ForCausalLM, directory, fp8_key: str | None = None
) -> str:
    """Save the text weights with the key layout of the multimodal checkpoint.

    A model with tied embeddings has no lm_head tensor in the checkpoint.
    ``fp8_key`` names one checkpoint tensor to store in FP8.
    """
    state = {}
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight" and model.config.tie_word_embeddings:
            continue
        if name.startswith("model."):
            name = "model.language_model." + name.removeprefix("model.")
        state[name] = tensor.contiguous()
    if fp8_key is not None:
        state[fp8_key] = state[fp8_key].to(torch.float8_e4m3fn)
    save_file(state, str(directory / "model.safetensors"))
    return str(directory)


def _attention_metadata(specs, slots: list[int], cold: bool) -> dict:
    return {
        specs[0].name: {
            "block_table_tensor": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_read_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_write_page_ids": torch.tensor([[2]], dtype=torch.int32),
            "recurrent_state_cold": torch.tensor([cold]),
            "recurrent_state_write_valid_mask": torch.tensor([[True]]),
        },
        specs[1].name: {
            "block_table_tensor": torch.tensor([[1, 2]], dtype=torch.int32),
            "slot_mapping": torch.tensor(slots),
            "block_size": BLOCK_SIZE,
        },
    }


def _use_the_plain_loader(monkeypatch):
    # The pipelined loader needs a process group. The plain loader applies the
    # same mappings and weight loaders.
    monkeypatch.setattr(
        SafetensorsCheckpoint,
        "load_sharded_pipelined",
        SafetensorsCheckpoint.load_sharded,
    )


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_tiny_model_matches_the_hf_text_model_in_fp32(tmp_path, monkeypatch, tied):
    # This test checks the FP32 torch path. Keep that path in simulator runs.
    monkeypatch.setattr(model_bf16, "can_run_kernel", lambda *_args, **_kwargs: False)
    _use_the_plain_loader(monkeypatch)
    hf_config = _hf_config(tied)
    hf_model = _hf_model(hf_config)
    checkpoint = _save_multimodal_layout(hf_model, tmp_path)

    config = Qwen3_5TextConfig.from_configs(hf_config)
    assert config.torch_dtype is torch.float32
    model = model_bf16.Qwen3_5ForCausalLM(config, neuron_config=None)
    model.load_weights(checkpoint, torch.device("cpu"), None)

    specs = model.get_kv_spec().layers
    _raw, recurrent_views = _state_views(specs[0])
    key = torch.zeros(4, 1, BLOCK_SIZE, config.head_dim)
    model.bind_kv_cache(
        {
            specs[0].name: recurrent_views,
            specs[1].name: [key, torch.zeros_like(key)],
        }
    )

    torch.manual_seed(46)
    tokens = torch.randint(0, hf_config.vocab_size, (len(TEXT_POSITIONS),))
    rotary = torch.tensor(MROPE_POSITIONS)
    # Block 1 holds slots 4 to 7, and block 2 holds slots 8 to 11.
    slots = [BLOCK_SIZE + position for position in TEXT_POSITIONS]
    with torch.no_grad():
        prefill = model(
            input_ids=tokens[:PREFILL_TOKENS],
            positions=torch.tensor(TEXT_POSITIONS[:PREFILL_TOKENS]),
            rotary_position_ids=rotary[:, :PREFILL_TOKENS],
            attn_metadata=_attention_metadata(specs, slots[:PREFILL_TOKENS], True),
            sampling_positions=torch.tensor([PREFILL_TOKENS - 1]),
        )
        decode = model(
            input_ids=tokens[PREFILL_TOKENS:],
            positions=torch.tensor(TEXT_POSITIONS[PREFILL_TOKENS:]),
            rotary_position_ids=rotary[:, PREFILL_TOKENS:],
            attn_metadata=_attention_metadata(specs, slots[PREFILL_TOKENS:], False),
            sampling_positions=torch.tensor([0]),
        )
        position_ids = torch.cat([torch.tensor([TEXT_POSITIONS]), rotary]).unsqueeze(1)
        expected = hf_model(
            input_ids=tokens.unsqueeze(0),
            position_ids=position_ids,
            use_cache=False,
        ).logits[0]

    torch.testing.assert_close(
        prefill[0], expected[PREFILL_TOKENS - 1], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        decode[0], expected[PREFILL_TOKENS], atol=ATOL, rtol=RTOL
    )


def test_load_weights_rejects_an_fp8_tensor_of_a_bf16_checkpoint(tmp_path, monkeypatch):
    # The loader casts each tensor to its parameter dtype, so the FP8 check must
    # read the stored dtype. Use the real loader and a real file.
    _use_the_plain_loader(monkeypatch)
    hf_config = _hf_config()
    checkpoint = _save_multimodal_layout(
        _hf_model(hf_config),
        tmp_path,
        fp8_key="model.language_model.layers.1.mlp.up_proj.weight",
    )
    model = model_bf16.Qwen3_5ForCausalLM(
        Qwen3_5TextConfig.from_configs(hf_config), neuron_config=None
    )

    with pytest.raises(ValueError, match="stores 1 tensors in FP8"):
        model.load_weights(checkpoint, torch.device("cpu"), None)
