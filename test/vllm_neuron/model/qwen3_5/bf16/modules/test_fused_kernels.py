# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 BF16 fused-kernel tests with the NKI simulator.

Each test runs one BF16 projection on its NKI kernel and again on its torch
path, with the same weights, at the TP 8 shapes of Qwen3.8-27B:
  - Attention q (with the gate), k and v: one prefill QKV kernel call on the
    joined weight. A second case also applies the input RMSNorm in the kernel.
  - Gated DeltaNet in_proj_qkv and in_proj_z: one prefill QKV kernel call on the
    joined weight, with and without the input RMSNorm.
  - A QKV call that NF.qkv_proj sends to qkv_tkg runs in torch, because
    qkv_tkg adds its partial sums in BF16.
  - Attention o_proj and Gated DeltaNet out_proj: the o_proj kernel.
  - MLP: the MLP kernel, with the post-attention RMSNorm.

Each test also checks the [qwen3_5] log, so a silent torch fallback fails it.

Run with:
    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 pytest \\
        test/vllm_neuron/model/qwen3_5/bf16/modules/test_fused_kernels.py -v
"""

import os
from contextlib import contextmanager

import pytest
import torch

from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig

SEED = 42
TP = 8
requires_simulator = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="Set NKI_SIMULATOR=1 to run the NKI simulator cases.",
)


@pytest.fixture(autouse=True)
def platform_override(monkeypatch):
    """Give the kernels a platform target on a host with no Neuron device."""
    if "NEURON_PLATFORM_TARGET_OVERRIDE" not in os.environ:
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")


# Qwen3.8-27B geometry. With TP 8: I/rank 2176, 3 heads of 256 per rank.
CONFIG = dict(
    hidden_size=5120,
    intermediate_size=17408,
    num_hidden_layers=4,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    torch_dtype=torch.bfloat16,
)
# The BF16 kernels and the torch path round differently in BF16.
MAX_REL = 0.02


class _StubGroup:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.rank_in_group = 0
        self.device_group = None

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # The test runs the shard of one rank.
        return x


@contextmanager
def stub_tp_group(world_size: int):
    from vllm_neuron.model.qwen3_5 import model_bf16

    original = model_bf16.get_tp_group
    model_bf16.get_tp_group = lambda: _StubGroup(world_size)
    try:
        yield
    finally:
        model_bf16.get_tp_group = original


@contextmanager
def torch_path():
    """Force the torch path of model_bf16 for the duration of the block."""
    from vllm_neuron.model.qwen3_5 import model_bf16

    original = model_bf16.can_run_kernel
    model_bf16.can_run_kernel = lambda *args, **kwargs: False
    try:
        yield
    finally:
        model_bf16.can_run_kernel = original


@pytest.fixture
def paths(monkeypatch):
    """Record each (op, path) decision that model_bf16 logs."""
    from vllm_neuron.model.qwen3_5 import model_bf16

    seen: list[tuple[str, str]] = []
    original = model_bf16._log_kernel_path

    def record(op: str, path: str, detail: str) -> None:
        seen.append((op, path))
        original(op, path, detail)

    monkeypatch.setattr(model_bf16, "_log_kernel_path", record)
    return seen


def _fill(weight: torch.nn.Parameter, gen: torch.Generator) -> None:
    """Give a weight values that keep the outputs O(1)."""
    weight.data = (
        torch.randn(weight.shape, generator=gen) / weight.shape[0] ** 0.5
    ).to(torch.bfloat16)


def _norm(cfg, gen):
    from vllm_neuron.model.qwen3_5.model_bf16 import Qwen3_5RMSNorm

    norm = Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, torch.bfloat16)
    norm.weight.data = (torch.randn(cfg.hidden_size, generator=gen) * 0.1).to(
        torch.bfloat16
    )
    return norm


def _rel_l2(got: torch.Tensor, expected: torch.Tensor) -> float:
    got, expected = got.float(), expected.float()
    return ((got - expected).norm() / expected.norm()).item()


def _run_both(fn, paths, op: str, kernel: str):
    """Return (kernel output, torch output) and check the logged paths."""
    with torch_path():
        expected = fn()
    paths.clear()
    got = fn()
    assert paths == [(op, kernel)], paths
    return got, expected


# NF.qkv_proj runs qkv_tkg when S <= 96 and B * S <= 128, and qkv_cte otherwise.
PREFILL_SHAPES = [
    pytest.param((1, 256), id="prefill-S256"),
    pytest.param((2, 128), id="prefill-B2-S128"),
    pytest.param((3, 48), id="cte-B3-S48"),
]
DECODE_SHAPES = [
    pytest.param((1, 1), id="decode-B1"),
    pytest.param((128, 1), id="decode-B128"),
    pytest.param((2, 64), id="tkg-B2-S64"),
    pytest.param((1, 96), id="tkg-S96"),
]
NORM_CASES = pytest.mark.parametrize(
    "with_norm", [False, True], ids=["no-norm", "fused-norm"]
)


@requires_simulator
@NORM_CASES
@pytest.mark.parametrize("shape", PREFILL_SHAPES)
def test_attention_qkv_kernel_matches_torch(shape, with_norm, paths):
    from vllm_neuron.model.qwen3_5.model_bf16 import Qwen3_5Attention

    gen = torch.Generator().manual_seed(SEED)
    cfg = Qwen3_5TextConfig(**CONFIG)
    with stub_tp_group(TP):
        attn = Qwen3_5Attention(cfg, 3)
    _fill(attn.qkv_proj_weight, gen)
    x = torch.randn(*shape, cfg.hidden_size, generator=gen).to(torch.bfloat16)
    norm = _norm(cfg, gen) if with_norm else None
    kernel = "NKI qkv + norm" if with_norm else "NKI qkv"

    got, expected = _run_both(
        lambda: torch.cat(attn._project_qkv(x, norm), dim=-1),
        paths,
        "Attention qkv",
        kernel,
    )
    assert got.shape == expected.shape == (*shape, attn.qkv_proj_weight.shape[1])
    assert _rel_l2(got, expected) < MAX_REL


@requires_simulator
@NORM_CASES
@pytest.mark.parametrize("shape", PREFILL_SHAPES)
def test_gdn_inputs_kernel_matches_torch(shape, with_norm, paths):
    from vllm_neuron.model.qwen3_5.model_bf16 import Qwen3_5GatedDeltaNet

    gen = torch.Generator().manual_seed(SEED)
    cfg = Qwen3_5TextConfig(**CONFIG)
    with stub_tp_group(TP):
        gdn = Qwen3_5GatedDeltaNet(cfg, 0)
    for weight in (gdn.in_proj_qkvz_weight, gdn.in_proj_a_weight, gdn.in_proj_b_weight):
        _fill(weight, gen)
    x = torch.randn(*shape, cfg.hidden_size, generator=gen).to(torch.bfloat16)
    norm = _norm(cfg, gen) if with_norm else None
    kernel = "NKI qkv + norm" if with_norm else "NKI qkv"

    def run():
        # in_proj_a and in_proj_b run in torch on both paths.
        return torch.cat(gdn._project_inputs(x, norm), dim=-1)

    got, expected = _run_both(run, paths, "GDN in_proj", kernel)
    width = gdn.in_proj_qkvz_weight.shape[1] + 2 * gdn.in_proj_a_weight.shape[1]
    assert got.shape == expected.shape == (*shape, width)
    assert _rel_l2(got, expected) < MAX_REL


@requires_simulator
@NORM_CASES
@pytest.mark.parametrize("shape", DECODE_SHAPES)
@pytest.mark.parametrize("mixer", ["attention", "gdn"])
def test_decode_size_qkv_runs_in_torch(mixer, shape, with_norm, paths):
    """qkv_tkg doubles the BF16 rounding error, so these calls stay in torch."""
    from vllm_neuron.model.qwen3_5.model_bf16 import (
        Qwen3_5Attention,
        Qwen3_5GatedDeltaNet,
    )

    gen = torch.Generator().manual_seed(SEED)
    cfg = Qwen3_5TextConfig(**CONFIG)
    with stub_tp_group(TP):
        if mixer == "attention":
            module, op = Qwen3_5Attention(cfg, 3), "Attention qkv"
            weights = (module.qkv_proj_weight,)

            def run(x, norm):
                return torch.cat(module._project_qkv(x, norm), dim=-1)

        else:
            module, op = Qwen3_5GatedDeltaNet(cfg, 0), "GDN in_proj"
            weights = (
                module.in_proj_qkvz_weight,
                module.in_proj_a_weight,
                module.in_proj_b_weight,
            )

            def run(x, norm):
                return torch.cat(module._project_inputs(x, norm), dim=-1)

    for weight in weights:
        _fill(weight, gen)
    x = torch.randn(*shape, cfg.hidden_size, generator=gen).to(torch.bfloat16)
    norm = _norm(cfg, gen) if with_norm else None

    got, expected = _run_both(lambda: run(x, norm), paths, op, "torch")
    assert torch.equal(got, expected)


@requires_simulator
@pytest.mark.parametrize("mixer", ["attention", "gdn"])
@pytest.mark.parametrize("tokens", [1, 256])
def test_output_projection_kernel_matches_torch(tokens, mixer, paths):
    from vllm_neuron.model.qwen3_5.model_bf16 import (
        Qwen3_5Attention,
        Qwen3_5GatedDeltaNet,
    )

    gen = torch.Generator().manual_seed(SEED)
    cfg = Qwen3_5TextConfig(**CONFIG)
    with stub_tp_group(TP):
        if mixer == "attention":
            module, weight, op = (
                Qwen3_5Attention(cfg, 3),
                "o_proj_weight",
                "Attention o_proj",
            )
        else:
            module, weight, op = (
                Qwen3_5GatedDeltaNet(cfg, 0),
                "out_proj_weight",
                "GDN out_proj",
            )
    _fill(getattr(module, weight), gen)
    width = getattr(module, weight).shape[0]
    x = torch.randn(1, tokens, width, generator=gen).to(torch.bfloat16)

    expected_path = (
        "torch FP32" if mixer == "attention" and tokens == 1 else "NKI o_proj"
    )
    got, expected = _run_both(
        lambda: module._project_output(x),
        paths,
        op,
        expected_path,
    )
    assert got.shape == expected.shape == (1, tokens, cfg.hidden_size)
    assert _rel_l2(got, expected) < MAX_REL


@requires_simulator
@pytest.mark.parametrize(
    "tokens",
    [
        pytest.param(1, id="decode-T1"),
        pytest.param(8, id="decode-T8"),
        pytest.param(256, id="prefill-T256"),
    ],
)
def test_mlp_kernel_with_the_norm_matches_torch(tokens, paths):
    from vllm_neuron.model.qwen3_5.model_bf16 import Qwen3_5MLP

    gen = torch.Generator().manual_seed(SEED)
    cfg = Qwen3_5TextConfig(**CONFIG)
    with stub_tp_group(TP):
        mlp = Qwen3_5MLP(cfg)
    for weight in (mlp.gate_proj_weight, mlp.up_proj_weight, mlp.down_proj_weight):
        _fill(weight, gen)
    norm = _norm(cfg, gen)
    hidden = torch.randn(1, tokens, cfg.hidden_size, generator=gen).to(torch.bfloat16)

    got, expected = _run_both(lambda: mlp(hidden, norm), paths, "MLP", "NKI mlp + norm")
    assert got.shape == expected.shape == hidden.shape
    assert _rel_l2(got, expected) < MAX_REL
