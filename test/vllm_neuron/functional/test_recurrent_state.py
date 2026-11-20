# SPDX-License-Identifier: Apache-2.0
"""Test model-agnostic recurrent state page access."""

import os
from importlib import import_module
from unittest import mock

import numpy as np
import pytest
import torch
from torch._dynamo.testing import CompileCounter
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_neuron import envs
from vllm_neuron.functional.recurrent_state import (
    mamba_state_row_width,
    mamba_state_rows,
    prepare_mamba_state_cache_access,
    read_cache_rows,
    read_mamba_state_pages,
    read_mamba_state_pages_grouped,
    write_cache_rows,
    write_mamba_state_pages,
    write_mamba_state_pages_grouped,
)
from vllm_neuron.model.kv_cache import (
    carve_mamba_state_views,
    prepare_mamba_state_cache,
)

SEED = 42
requires_simulator = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="Set NKI_SIMULATOR=1 to run the NKI simulator tests.",
)


def _state_case(num_pages: int = 5):
    """Create two state views in one padded BF16 page pool."""
    torch.manual_seed(SEED)
    conv_elements = 3 * 8 * 3
    recurrent_elements = 2 * 4 * 4
    page_elements = conv_elements + recurrent_elements + 24
    base = torch.randn(num_pages * page_elements).bfloat16()
    conv = torch.as_strided(
        base,
        (num_pages, 3, 8, 3),
        (page_elements, 24, 3, 1),
        0,
    )
    recurrent = torch.as_strided(
        base,
        (num_pages, 2, 4, 4),
        (page_elements, 16, 4, 1),
        conv_elements,
    )
    bases, layout = prepare_mamba_state_cache([conv, recurrent])
    return base, conv, recurrent, bases, layout


def test_prepare_mamba_state_cache_access_prewarms_row_kernels():
    base = torch.zeros(16, dtype=torch.bfloat16)
    view = torch.as_strided(base, (2, 4), (8, 1))

    with mock.patch(
        "vllm_neuron.functional.recurrent_state.prewarm_mamba_state_access"
    ) as prewarm:
        bases, layout = prepare_mamba_state_cache_access([view])

    prewarm.assert_called_once_with(bases, layout)


def test_prewarm_uses_state_read_widths_and_one_shared_write_width():
    recurrent_state = import_module("vllm_neuron.functional.recurrent_state")
    base = torch.zeros(64, dtype=torch.bfloat16)
    first = torch.as_strided(base, (4, 8), (16, 1), 0)
    second = torch.as_strided(base, (4, 4), (16, 1), 8)
    bases, layout = prepare_mamba_state_cache((first, second))

    with (
        mock.patch.object(recurrent_state, "_READ_SHIMS", {}),
        mock.patch.object(recurrent_state, "_WRITE_SHIMS", {}),
        mock.patch.object(recurrent_state, "_can_use_row_kernel", return_value=True),
        mock.patch.object(
            recurrent_state,
            "_build_row_shim",
            side_effect=lambda width, read=False: (width, read),
        ) as build,
    ):
        recurrent_state.prewarm_mamba_state_access(bases, layout)

    assert build.call_args_list == [
        mock.call(8, read=True),
        mock.call(4, read=True),
        mock.call(4),
    ]


@pytest.mark.parametrize("lnc", [1, 2])
def test_row_kernel_grid_uses_the_device_lnc(monkeypatch, lnc):
    recurrent_state = import_module("vllm_neuron.functional.recurrent_state")
    monkeypatch.setattr(envs, "VLLM_NEURON_CPU_MODE", False)
    monkeypatch.setattr(recurrent_state, "get_logical_nc_config", lambda: lnc)

    assert recurrent_state._row_kernel_grid() == lnc


def test_read_mamba_state_pages_selects_scheduler_pages():
    _, conv, recurrent, bases, layout = _state_case()
    table = torch.tensor([[3], [1]], dtype=torch.int32)

    actual_conv, actual_recurrent = read_mamba_state_pages(bases, layout, table)

    torch.testing.assert_close(actual_conv, conv[[3, 1]])
    torch.testing.assert_close(actual_recurrent, recurrent[[3, 1]])


def test_read_mamba_state_pages_zeroes_invalid_and_cold_requests():
    _, conv, recurrent, bases, layout = _state_case()
    table = torch.tensor([[2], [-1], [0], [9]], dtype=torch.int32)
    cold = torch.tensor([True, False, False, False])

    actual_conv, actual_recurrent = read_mamba_state_pages(
        bases,
        layout,
        table,
        cold=cold,
    )

    assert torch.count_nonzero(actual_conv) == 0
    assert torch.count_nonzero(actual_recurrent) == 0
    assert torch.count_nonzero(conv[2]) > 0
    assert torch.count_nonzero(recurrent[2]) > 0


def test_write_mamba_state_pages_updates_only_live_pages():
    base, conv, recurrent, bases, layout = _state_case()
    before = base.clone()
    table = torch.tensor([[3], [0], [-1], [7]], dtype=torch.int32)
    next_conv = torch.full((4, 3, 8, 3), 11, dtype=torch.float32)
    next_recurrent = torch.full((4, 2, 4, 4), 13, dtype=torch.float32)

    write_mamba_state_pages(
        bases,
        layout,
        table,
        (next_conv, next_recurrent),
    )

    torch.testing.assert_close(conv[3], next_conv[0].bfloat16())
    torch.testing.assert_close(recurrent[3], next_recurrent[0].bfloat16())
    page = conv.stride(0)
    torch.testing.assert_close(base[: 3 * page], before[: 3 * page])
    torch.testing.assert_close(base[4 * page :], before[4 * page :])


def test_read_then_write_reuses_unique_owned_pages():
    _, conv, recurrent, bases, layout = _state_case()
    table = torch.tensor([[1], [3]], dtype=torch.int32)
    expected_old_conv = conv[[1, 3]].clone()
    expected_old_recurrent = recurrent[[1, 3]].clone()
    old_conv, old_recurrent = read_mamba_state_pages(bases, layout, table)
    next_conv = old_conv.float() + 5
    next_recurrent = old_recurrent.float() + 7

    write_mamba_state_pages(
        bases,
        layout,
        table,
        (next_conv, next_recurrent),
    )
    actual_conv, actual_recurrent = read_mamba_state_pages(bases, layout, table)

    torch.testing.assert_close(old_conv, expected_old_conv)
    torch.testing.assert_close(old_recurrent, expected_old_recurrent)
    torch.testing.assert_close(actual_conv, next_conv.bfloat16())
    torch.testing.assert_close(actual_recurrent, next_recurrent.bfloat16())


def test_mixed_dtype_pages_use_one_shared_carrier():
    spec = MambaSpec(
        block_size=10,
        shapes=((4, 3), (2, 4, 4)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=160,
    )
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    conv, recurrent = carve_mamba_state_views(raw, spec)
    conv[2].fill_(3)
    recurrent[2].fill_(7)
    bases, layout = prepare_mamba_state_cache_access((conv, recurrent))
    table = torch.tensor([[2]], dtype=torch.int32)
    recurrent_state = import_module("vllm_neuron.functional.recurrent_state")

    with mock.patch.object(
        recurrent_state,
        "read_cache_rows",
        wraps=recurrent_state.read_cache_rows,
    ) as read_rows:
        old_conv, old_recurrent = read_mamba_state_pages(bases, layout, table)
    with mock.patch.object(
        recurrent_state,
        "write_cache_rows",
        wraps=recurrent_state.write_cache_rows,
    ) as write_rows:
        write_mamba_state_pages(
            bases,
            layout,
            table,
            (old_conv + 8, old_recurrent + 6),
        )

    assert [base.dtype for base in bases] == [torch.bfloat16]
    assert read_rows.call_count == 1
    assert write_rows.call_count == 1
    assert torch.all(conv[2] == 11)
    assert torch.all(recurrent[2] == 13)


def test_grouped_pages_use_one_read_and_write_for_a_shared_carrier():
    _, conv, recurrent, bases, layout = _state_case()
    recurrent_state = import_module("vllm_neuron.functional.recurrent_state")
    first_conv = torch.full((1, 3, 8, 3), 11, dtype=torch.float32)
    first_recurrent = torch.full((1, 2, 4, 4), 13, dtype=torch.float32)
    second_conv = torch.full((1, 3, 8, 3), 17, dtype=torch.float32)
    second_recurrent = torch.full((1, 2, 4, 4), 19, dtype=torch.float32)

    with mock.patch.object(
        recurrent_state,
        "write_cache_rows",
        wraps=recurrent_state.write_cache_rows,
    ) as write_rows:
        write_mamba_state_pages_grouped(
            bases,
            (
                (
                    layout,
                    torch.tensor([[1]], dtype=torch.int32),
                    (first_conv, first_recurrent),
                ),
                (
                    layout,
                    torch.tensor([[3]], dtype=torch.int32),
                    (second_conv, second_recurrent),
                ),
            ),
        )

    with mock.patch.object(
        recurrent_state,
        "read_cache_rows",
        wraps=recurrent_state.read_cache_rows,
    ) as read_rows:
        values = read_mamba_state_pages_grouped(
            bases,
            (
                (layout, torch.tensor([[1]]), None),
                (layout, torch.tensor([[3]]), None),
            ),
        )

    assert write_rows.call_count == 1
    assert read_rows.call_count == 1
    torch.testing.assert_close(values[0][0], first_conv.bfloat16())
    torch.testing.assert_close(values[0][1], first_recurrent.bfloat16())
    torch.testing.assert_close(values[1][0], second_conv.bfloat16())
    torch.testing.assert_close(values[1][1], second_recurrent.bfloat16())
    torch.testing.assert_close(conv[1], first_conv[0].bfloat16())
    torch.testing.assert_close(recurrent[3], second_recurrent[0].bfloat16())


@pytest.mark.parametrize("group_count", [1, 2, 3])
def test_grouped_state_access_preserves_group_and_page_order(group_count):
    _, conv, recurrent, bases, layout = _state_case()
    updates = []
    expected = {}
    for group_index in range(group_count):
        page = group_index + 1
        conv_value = torch.full(
            (1, 3, 8, 3),
            11 + group_index,
            dtype=torch.float32,
        )
        recurrent_value = torch.full(
            (1, 2, 4, 4),
            21 + group_index,
            dtype=torch.float32,
        )
        updates.append(
            (
                layout,
                torch.tensor([[page]], dtype=torch.int32),
                (conv_value, recurrent_value),
            )
        )
        expected[page] = (conv_value.bfloat16(), recurrent_value.bfloat16())

    write_mamba_state_pages_grouped(bases, updates)
    read_pages = list(range(group_count, 0, -1))
    values = read_mamba_state_pages_grouped(
        bases,
        [
            (
                layout,
                torch.tensor([[page]], dtype=torch.int32),
                None,
            )
            for page in read_pages
        ],
    )

    for group_index, page in enumerate(read_pages):
        expected_conv, expected_recurrent = expected[page]
        torch.testing.assert_close(values[group_index][0], expected_conv)
        torch.testing.assert_close(values[group_index][1], expected_recurrent)
        torch.testing.assert_close(conv[page], expected_conv[0])
        torch.testing.assert_close(recurrent[page], expected_recurrent[0])


def test_state_write_stores_each_speculative_checkpoint():
    _, conv, recurrent, bases, layout = _state_case(num_pages=7)
    table = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int32)
    next_conv = torch.arange(2 * 3 * 3 * 8 * 3, dtype=torch.float32).reshape(
        2, 3, 3, 8, 3
    )
    next_recurrent = torch.arange(
        2 * 3 * 2 * 4 * 4,
        dtype=torch.float32,
    ).reshape(2, 3, 2, 4, 4)

    write_mamba_state_pages(
        bases,
        layout,
        table,
        (next_conv, next_recurrent),
    )

    for request in range(2):
        for checkpoint in range(3):
            page = int(table[request, checkpoint])
            if page == 0:
                continue
            torch.testing.assert_close(
                conv[page],
                next_conv[request, checkpoint].bfloat16(),
            )
            torch.testing.assert_close(
                recurrent[page],
                next_recurrent[request, checkpoint].bfloat16(),
            )


def test_multi_column_read_returns_what_a_write_stores():
    _, conv, recurrent, bases, layout = _state_case(num_pages=7)
    table = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int32)
    next_conv = torch.randn(2, 3, 3, 8, 3)
    next_recurrent = torch.randn(2, 3, 2, 4, 4)
    write_mamba_state_pages(bases, layout, table, (next_conv, next_recurrent))

    read_conv, read_recurrent = read_mamba_state_pages(bases, layout, table)
    cold_conv, _ = read_mamba_state_pages(
        bases, layout, table, cold=torch.tensor([True, False])
    )

    assert read_conv.shape == next_conv.shape
    assert read_recurrent.shape == next_recurrent.shape
    written = (table > 0).reshape(2, 3, 1, 1, 1)
    torch.testing.assert_close(
        read_conv, torch.where(written, next_conv, 0).to(conv.dtype)
    )
    torch.testing.assert_close(
        read_recurrent, torch.where(written, next_recurrent, 0).to(recurrent.dtype)
    )
    # One cold value per request resets every column of that request.
    assert not cold_conv[0].any()
    torch.testing.assert_close(cold_conv[1], read_conv[1])


def test_state_write_accepts_an_explicit_single_checkpoint_axis():
    _, conv, recurrent, bases, layout = _state_case(num_pages=4)
    table = torch.tensor([[1], [2]], dtype=torch.int32)
    next_conv = torch.arange(2 * 3 * 8 * 3, dtype=torch.float32).reshape(2, 1, 3, 8, 3)
    next_recurrent = torch.arange(
        2 * 2 * 4 * 4,
        dtype=torch.float32,
    ).reshape(2, 1, 2, 4, 4)

    write_mamba_state_pages(
        bases,
        layout,
        table,
        (next_conv, next_recurrent),
    )

    torch.testing.assert_close(conv[1], next_conv[0, 0].bfloat16())
    torch.testing.assert_close(conv[2], next_conv[1, 0].bfloat16())
    torch.testing.assert_close(recurrent[1], next_recurrent[0, 0].bfloat16())
    torch.testing.assert_close(recurrent[2], next_recurrent[1, 0].bfloat16())


def test_recurrent_state_access_compiles_to_one_cpu_graph():
    torch._dynamo.reset()
    _, _, _, bases, layout = _state_case()
    counter = CompileCounter()

    def step(block_table, cold, next_conv, next_recurrent):
        old = read_mamba_state_pages(bases, layout, block_table, cold=cold)
        write_mamba_state_pages(
            bases,
            layout,
            block_table,
            (next_conv, next_recurrent),
        )
        return old

    compiled = torch.compile(step, backend=counter, fullgraph=True)
    next_conv = torch.ones(1, 3, 8, 3)
    next_recurrent = torch.ones(1, 2, 4, 4)
    for page in (1, 4):
        compiled(
            torch.tensor([[page]], dtype=torch.int32),
            torch.tensor([False]),
            next_conv,
            next_recurrent,
        )

    assert counter.frame_count == 1


def test_row_helpers_cover_page_payload_and_skip_invalid_rows():
    shape, stride, offset = (5, 3, 4), (20, 4, 1), 8
    width = mamba_state_row_width([(shape, stride, offset)], itemsize=2)

    rows = mamba_state_rows(
        torch.tensor([2, 4, 1]),
        torch.tensor([True, False, True]),
        shape,
        stride,
        offset,
        width,
    )

    assert width == 4
    assert rows.tolist() == [12, 13, 14, -1, -1, -1, 7, 8, 9]


def test_cache_row_helpers_read_and_write_only_valid_rows():
    torch.manual_seed(SEED)
    base = torch.randn(6, 4, 8)
    before = base.clone()
    index = torch.tensor([3, -1, 23, 24, 0])
    rows = torch.randn(5, 8)

    write_cache_rows(base, index, rows, width=8)
    actual = read_cache_rows(base, index, width=8)

    torch.testing.assert_close(base.view(-1, 8)[3], rows[0])
    torch.testing.assert_close(base.view(-1, 8)[23], rows[2])
    torch.testing.assert_close(base.view(-1, 8)[0], rows[4])
    torch.testing.assert_close(base.view(-1, 8)[1], before.view(-1, 8)[1])
    torch.testing.assert_close(actual[0], rows[0])
    torch.testing.assert_close(actual[2], rows[2])
    torch.testing.assert_close(actual[4], rows[4])
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.count_nonzero(actual[3]) == 0


def test_invalid_cache_row_write_does_not_restore_row_zero():
    base = torch.arange(8, dtype=torch.float32)
    rows = torch.tensor([[100.0, 101.0], [900.0, 901.0]])

    write_cache_rows(
        base,
        torch.tensor([0, -1]),
        rows,
        width=2,
    )

    torch.testing.assert_close(base[:2], rows[0])
    torch.testing.assert_close(base[2:], torch.arange(2, 8, dtype=torch.float32))


def test_invalid_metadata_reports_the_required_fix():
    _, _, _, bases, layout = _state_case()

    with pytest.raises(ValueError, match="Provide one cold value for each request"):
        read_mamba_state_pages(
            bases,
            layout,
            torch.tensor([[1], [2]]),
            cold=torch.tensor([False]),
        )
    with pytest.raises(ValueError, match="Provide one state"):
        write_mamba_state_pages(
            bases,
            layout,
            torch.tensor([[1]]),
            (torch.zeros(1, 3, 8, 3),),
        )
    with pytest.raises(ValueError, match="Use an accepted state shape"):
        write_mamba_state_pages(
            bases,
            layout,
            torch.tensor([[1]]),
            (
                torch.zeros(2, 3, 8, 3),
                torch.zeros(1, 2, 4, 4),
            ),
        )


def test_read_row_kernel_does_not_alias_the_cache_carrier():
    recurrent_state = import_module("vllm_neuron.functional.recurrent_state")
    base = torch.zeros(8, 4)
    index = torch.tensor([1, 2], dtype=torch.int64)
    valid = torch.ones(2, dtype=torch.bool)
    kernel_rows = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)

    with (
        mock.patch.object(recurrent_state, "_row_shim_for", return_value=object()),
        mock.patch.object(
            recurrent_state,
            "_call_row_kernel",
            return_value=kernel_rows,
        ) as call,
    ):
        actual = recurrent_state._read_rows_kernel(
            base,
            num_rows=8,
            index=index,
            valid=valid,
            width=4,
        )

    assert call.call_args.args[1] == {}
    torch.testing.assert_close(actual, kernel_rows[:2])


@requires_simulator
def test_simulator_row_kernels_skip_invalid_rows():
    import nki

    from vllm_neuron._kernels.state_cache.recurrent_state_rows import (
        read_rows,
        write_rows_inplace,
    )

    rows, width = 256, 8
    base = np.arange(rows * width, dtype=np.float32).reshape(rows, width)
    index = np.full((128, 1), rows + 1, dtype=np.int32)
    index[:3, 0] = np.array([5, rows + 1, 7], dtype=np.int32)
    updates = np.zeros((128, width), dtype=np.float32)
    updates[0] = 11
    updates[2] = 13
    expected = base.copy()
    expected[5] = 11
    expected[7] = 13

    written = nki.simulate(write_rows_inplace)(
        base.copy(),
        index,
        updates,
        width=width,
    )
    read = nki.simulate(read_rows)(
        expected.copy(),
        index,
        width=width,
    )

    np.testing.assert_array_equal(np.asarray(written), expected)
    np.testing.assert_array_equal(np.asarray(read)[0], expected[5])
    np.testing.assert_array_equal(np.asarray(read)[2], expected[7])
    assert np.count_nonzero(np.asarray(read)[1]) == 0
