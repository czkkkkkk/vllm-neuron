# SPDX-License-Identifier: Apache-2.0
"""On-device correctness test for shared recurrent-state row kernels."""

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_neuron import envs
from vllm_neuron.functional.recurrent_state import (
    prepare_mamba_state_cache_access,
    read_mamba_state_pages,
    write_mamba_state_pages,
)
from vllm_neuron.model.kv_cache import carve_mamba_state_views

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_neuron,
    pytest.mark.skipif(
        envs.VLLM_NEURON_CPU_MODE,
        reason="This test compiles and executes on a Neuron device.",
    ),
]

_CONV_STATE_SHAPE = (3, 1280)
_RECURRENT_STATE_SHAPE = (6, 128, 128)


def _state_spec() -> MambaSpec:
    """Return the Qwen3.8-27B per-rank GDN state layout at TP 8."""
    return MambaSpec(
        block_size=1,
        shapes=(_CONV_STATE_SHAPE, _RECURRENT_STATE_SHAPE),
        dtypes=(torch.bfloat16, torch.float32),
    )


class _RecurrentStateRoundTrip(torch.nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.layout = layout

    def forward(self, base, block_table, next_conv, next_recurrent):
        old_conv, old_recurrent = read_mamba_state_pages(
            (base,),
            self.layout,
            block_table,
        )
        write_mamba_state_pages(
            (base,),
            self.layout,
            block_table,
            (next_conv, next_recurrent),
        )
        conv, recurrent = read_mamba_state_pages(
            (base,),
            self.layout,
            block_table,
        )
        return old_conv, old_recurrent, conv, recurrent, base


@pytest.mark.parametrize(
    "_device_family",
    [
        pytest.param("trn2", id="trn2"),
        pytest.param("trn3", marks=pytest.mark.trn3, id="trn3"),
    ],
)
def test_recurrent_state_rows_compile_and_preserve_shared_carrier(
    _device_family: str,
) -> None:
    torch.manual_seed(42)
    batch = 6
    num_pages = batch + 3
    spec = _state_spec()
    raw = torch.zeros(num_pages * spec.page_size_bytes, dtype=torch.int8)
    initial_base = raw.view(torch.bfloat16).clone()

    block_table = torch.arange(1, batch + 1, dtype=torch.int32).reshape(batch, 1)
    block_table[-2, 0] = 0
    block_table[-1, 0] = num_pages + 7
    next_conv = torch.randn(batch, *_CONV_STATE_SHAPE, dtype=torch.bfloat16)
    next_recurrent = torch.randn(batch, *_RECURRENT_STATE_SHAPE, dtype=torch.float32)

    expected_raw = raw.clone()
    expected_conv_view, expected_recurrent_view = carve_mamba_state_views(
        expected_raw,
        spec,
    )
    expected_base = expected_raw.view(torch.bfloat16)
    expected_conv = torch.zeros_like(next_conv)
    expected_recurrent = torch.zeros_like(next_recurrent)
    for request, page in enumerate(block_table.reshape(-1).tolist()):
        if 0 < page < num_pages:
            expected_conv_view[page].copy_(next_conv[request])
            expected_recurrent_view[page].copy_(next_recurrent[request])
            expected_conv[request].copy_(next_conv[request])
            expected_recurrent[request].copy_(next_recurrent[request])

    device = torch.device("neuron:0")
    device_raw = raw.to(device)
    device_conv, device_recurrent = carve_mamba_state_views(device_raw, spec)
    bases, layout = prepare_mamba_state_cache_access((device_conv, device_recurrent))
    module = torch.compile(
        _RecurrentStateRoundTrip(layout).to(device),
        backend=envs.get_compile_backend_name(),
        fullgraph=True,
    )
    old_conv, old_recurrent, actual_conv, actual_recurrent, updated_base = module(
        bases[0],
        block_table.to(device),
        next_conv.to(device),
        next_recurrent.to(device),
    )

    old_conv = old_conv.cpu()
    old_recurrent = old_recurrent.cpu()
    actual_conv = actual_conv.cpu()
    actual_recurrent = actual_recurrent.cpu()
    updated_base = updated_base.cpu()
    aliased_input_base = bases[0].cpu()
    assert torch.equal(
        updated_base.view(torch.int16),
        aliased_input_base.view(torch.int16),
    )
    assert torch.count_nonzero(old_conv) == 0
    assert torch.count_nonzero(old_recurrent) == 0
    assert torch.equal(actual_conv.view(torch.int16), expected_conv.view(torch.int16))
    assert torch.equal(
        actual_recurrent.view(torch.int32),
        expected_recurrent.view(torch.int32),
    )
    assert torch.equal(
        updated_base.view(torch.int16),
        expected_base.view(torch.int16),
    )
    null_page_elements = spec.page_size_bytes // updated_base.element_size()
    assert torch.equal(
        updated_base[:null_page_elements].view(torch.int16),
        initial_base[:null_page_elements].view(torch.int16),
    )
