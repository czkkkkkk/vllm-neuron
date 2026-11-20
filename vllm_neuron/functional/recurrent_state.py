# SPDX-License-Identifier: Apache-2.0
"""Read and write selected pages of a recurrent state cache.

The public functions use the block table to read or write one state page for
each request. On Neuron, an in-place row kernel updates the contiguous cache
base. This operation avoids a copy and a writeback of every cache page.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Sequence

import torch
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.model.kv_cache import (
    MambaStateViewLayout,
    mamba_state_page_ids,
    prepare_mamba_state_cache,
)
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

_PARTITIONS = 128
_MAX_ROW_BYTES = 8192
_WRITE_SHIMS: dict[int, object] = {}
_READ_SHIMS: dict[int, object] = {}


def prepare_mamba_state_cache_access(
    state_views: Sequence[Tensor],
) -> tuple[list[Tensor], MambaStateViewLayout]:
    """Prepare recurrent state pages for model execution.

    This function keeps one carrier base for each allocation. It prepares the
    row kernels before graph capture. CPU execution does not build the kernels.

    Example:
        >>> base = torch.zeros(16, dtype=torch.bfloat16)
        >>> view = torch.as_strided(base, (2, 4), (8, 1))
        >>> bases, layout = prepare_mamba_state_cache_access([view])
        >>> (len(bases), len(layout))
        (1, 1)
    """
    bases, layout = prepare_mamba_state_cache(state_views)
    prewarm_mamba_state_access(bases, layout)
    return bases, layout


def read_mamba_state_pages(
    bases: Sequence[Tensor],
    layout: MambaStateViewLayout,
    block_table: Tensor,
    *,
    cold: Tensor | None = None,
) -> list[Tensor]:
    """Read one recurrent state page for each request and table column.

    An invalid request reads zeros. A padded request also reads zeros.
    ``cold`` resets the state for each selected request. A ``[B, 1]`` table
    returns ``[B, ...]`` states, and a ``[B, C]`` table returns
    ``[B, C, ...]`` states, the shapes that a write accepts.

    Example:
        >>> states = read_mamba_state_pages(  # doctest: +SKIP
        ...     bases, layout, block_table, cold=cold
        ... )
    """
    return read_mamba_state_pages_grouped(
        bases,
        ((layout, block_table, cold),),
    )[0]


def read_mamba_state_pages_grouped(
    bases: Sequence[Tensor],
    reads: Sequence[
        tuple[
            MambaStateViewLayout,
            Tensor,
            Tensor | None,
        ]
    ],
) -> list[list[Tensor]]:
    """Read logical layers with one row read for each physical base.

    Example:
        >>> states = read_mamba_state_pages_grouped(  # doctest: +SKIP
        ...     bases, [(layout, block_table, cold)]
        ... )
    """
    if not reads:
        return []
    records: dict[
        int,
        list[
            tuple[
                int,
                int,
                torch.dtype,
                tuple[int, ...],
                tuple[tuple[int, ...], tuple[int, ...], int],
                Tensor,
                Tensor,
            ]
        ],
    ] = {}
    values: list[list[Tensor | None]] = []
    output_shapes: list[tuple[int, ...]] = []
    for read_index, (layout, block_table, cold) in enumerate(reads):
        num_pages = _validate_layout(bases, layout)
        page_table = mamba_state_page_ids(block_table, num_pages)
        batch_size, columns = page_table.shape
        page_ids = page_table.reshape(-1)
        valid = page_ids > 0
        if cold is not None:
            cold_mask = cold.to(torch.bool)
            if cold_mask.numel() == batch_size and columns > 1:
                # One cold value per request applies to each of its columns.
                cold_mask = cold_mask.reshape(batch_size, 1).expand(batch_size, columns)
            if cold_mask.numel() != valid.numel():
                raise ValueError(
                    f"The cold mask has {cold_mask.numel()} values, but the "
                    f"page table has {batch_size} requests and {columns} "
                    "columns. Provide one cold value for each request, or "
                    "one for each request and column."
                )
            valid = valid & ~cold_mask.reshape(-1)
        output_shapes.append((batch_size,) if columns == 1 else (batch_size, columns))
        values.append([None] * len(layout))

        for state_index, (base_index, dtype, shape, stride, offset) in enumerate(
            layout
        ):
            base_layout = _state_layout_in_base(
                bases[base_index],
                dtype,
                shape,
                stride,
                offset,
            )
            records.setdefault(base_index, []).append(
                (
                    read_index,
                    state_index,
                    dtype,
                    shape,
                    base_layout,
                    page_ids,
                    valid,
                )
            )

    for base_index, items in records.items():
        base = bases[base_index]
        width = mamba_state_row_width(
            [base_layout for _, _, _, _, base_layout, _, _ in items],
            base.element_size(),
        )
        indices = [
            mamba_state_rows(page_ids, valid, *base_layout, width)
            for _, _, _, _, base_layout, page_ids, valid in items
        ]
        rows = read_cache_rows(base, torch.cat(indices), width=width)
        start = 0
        for (
            read_index,
            state_index,
            dtype,
            shape,
            _base_layout,
            page_ids,
            _valid,
        ), index in zip(items, indices, strict=True):
            count = index.shape[0]
            value = rows[start : start + count].reshape(page_ids.shape[0], -1)
            if value.dtype != dtype:
                value = value.view(dtype)
            values[read_index][state_index] = value.reshape(
                *output_shapes[read_index],
                *shape[1:],
            )
            start += count

    if any(item is None for group in values for item in group):
        raise RuntimeError(
            "A recurrent state read did not return each state. Check the state layout."
        )
    return [[item for item in group if item is not None] for group in values]


def write_mamba_state_pages(
    bases: Sequence[Tensor],
    layout: MambaStateViewLayout,
    block_table: Tensor,
    states: Sequence[Tensor],
) -> None:
    """Write one recurrent state page for each valid request.

    Valid page IDs must be unique across requests and checkpoint columns. The
    scheduler gives each update a unique writable page. A caller can read an
    owned page before it writes the same page.

    Example:
        >>> write_mamba_state_pages(  # doctest: +SKIP
        ...     bases, layout, block_table, states
        ... )
    """
    write_mamba_state_pages_grouped(
        bases,
        ((layout, block_table, states),),
    )


def write_mamba_state_pages_grouped(
    bases: Sequence[Tensor],
    updates: Sequence[
        tuple[
            MambaStateViewLayout,
            Tensor,
            Sequence[Tensor],
        ]
    ],
) -> None:
    """Write logical layers with one row update for each physical base.

    Valid page IDs must be unique across all updates in this call. The scheduler
    owns this allocation contract.

    Example:
        >>> write_mamba_state_pages_grouped(  # doctest: +SKIP
        ...     bases, [(layout, block_table, states)]
        ... )
    """
    if not updates:
        return
    writes: dict[
        int,
        list[
            tuple[
                Tensor,
                Tensor,
                torch.dtype,
                tuple[int, ...],
                tuple[int, ...],
                int,
                Tensor,
            ]
        ],
    ] = {}
    for layout, block_table, states in updates:
        num_pages = _validate_layout(bases, layout)
        if len(states) != len(layout):
            raise ValueError(
                f"The update has {len(states)} states, but the layout has "
                f"{len(layout)} states. Provide one state for each layout entry."
            )

        page_table = mamba_state_page_ids(block_table, num_pages)
        batch_size, checkpoint_count = page_table.shape
        page_ids = page_table.reshape(-1)
        valid = page_ids > 0
        for (base_index, dtype, shape, stride, offset), value in zip(
            layout, states, strict=True
        ):
            collapsed_shape = (batch_size, *shape[1:])
            checkpoint_shape = (batch_size, checkpoint_count, *shape[1:])
            accepted_shapes = (
                (collapsed_shape, checkpoint_shape)
                if checkpoint_count == 1
                else (checkpoint_shape,)
            )
            if tuple(value.shape) not in accepted_shapes:
                raise ValueError(
                    f"The state has shape {tuple(value.shape)}, but the accepted "
                    f"shapes are {accepted_shapes}. Use an accepted state shape."
                )
            value = value.reshape(page_ids.shape[0], *shape[1:])
            writes.setdefault(base_index, []).append(
                (page_ids, valid, dtype, shape, stride, offset, value)
            )

    for base_index, items in writes.items():
        base = bases[base_index]
        base_layouts = [
            _state_layout_in_base(base, dtype, shape, stride, offset)
            for _page_ids, _valid, dtype, shape, stride, offset, _value in items
        ]
        width = mamba_state_row_width(
            base_layouts,
            base.element_size(),
        )
        indices = []
        rows = []
        for (
            page_ids,
            valid,
            dtype,
            _shape,
            _stride,
            _offset,
            value,
        ), base_layout in zip(
            items,
            base_layouts,
            strict=True,
        ):
            indices.append(
                mamba_state_rows(
                    page_ids,
                    valid,
                    *base_layout,
                    width,
                )
            )
            packed = value.to(dtype).contiguous()
            if packed.dtype != base.dtype:
                packed = packed.view(base.dtype)
            rows.append(packed.reshape(-1, width))
        write_cache_rows(
            base,
            torch.cat(indices),
            torch.cat(rows),
            width=width,
        )


def prewarm_mamba_state_access(
    bases: Sequence[Tensor],
    layout: MambaStateViewLayout,
) -> None:
    """Build the row kernel trace shims for one state layout.

    Example:
        >>> prewarm_mamba_state_access(bases, layout)  # doctest: +SKIP
    """
    _validate_layout(bases, layout)
    grouped: dict[int, list[tuple[tuple[int, ...], tuple[int, ...], int]]] = {}
    for base_index, dtype, shape, stride, offset in layout:
        base = bases[base_index]
        base_layout = _state_layout_in_base(base, dtype, shape, stride, offset)
        grouped.setdefault(base_index, []).append(base_layout)
        read_width = mamba_state_row_width([base_layout], base.element_size())
        if not _can_use_row_kernel(base, read_width):
            continue
        if read_width not in _READ_SHIMS:
            _READ_SHIMS[read_width] = _build_row_shim(read_width, read=True)

    for base_index, base_layouts in grouped.items():
        base = bases[base_index]
        write_width = mamba_state_row_width(base_layouts, base.element_size())
        if _can_use_row_kernel(base, write_width) and write_width not in _WRITE_SHIMS:
            _WRITE_SHIMS[write_width] = _build_row_shim(write_width)


def mamba_state_rows(
    page_ids: Tensor,
    valid: Tensor,
    shape: Sequence[int],
    stride: Sequence[int],
    storage_offset: int,
    width: int,
) -> Tensor:
    """Return flat row indices for each selected state page.

    Example:
        >>> ids = torch.tensor([1])
        >>> valid = torch.tensor([True])
        >>> mamba_state_rows(ids, valid, (2, 4), (4, 1), 0, 2).tolist()
        [2, 3]
    """
    payload = _contiguous_payload(shape, stride)
    if payload % width or stride[0] % width or storage_offset % width:
        raise ValueError(
            f"The row width {width} does not divide the state page layout. "
            "Use a common row width for the state layout."
        )
    rows_per_page = payload // width
    first = (storage_offset + page_ids.to(torch.int64) * int(stride[0])) // width
    index = first.reshape(-1, 1) + torch.arange(
        rows_per_page,
        dtype=torch.int64,
        device=page_ids.device,
    ).reshape(1, -1)
    index = torch.where(
        valid.reshape(-1, 1),
        index,
        torch.full_like(index, -1),
    )
    return index.reshape(-1)


def mamba_state_row_width(
    layouts: Sequence[tuple[Sequence[int], Sequence[int], int]],
    itemsize: int,
) -> int:
    """Select the widest power-of-two row for all state layouts.

    Example:
        >>> mamba_state_row_width([((2, 4), (4, 1), 0)], itemsize=2)
        4
    """
    common = 0
    for shape, stride, offset in layouts:
        common = math.gcd(common, _contiguous_payload(shape, stride))
        common = math.gcd(common, int(stride[0]))
        common = math.gcd(common, int(offset))
    width = 1
    while common % (width * 2) == 0 and width * 2 * itemsize <= _MAX_ROW_BYTES:
        width *= 2
    return width


def write_cache_rows(
    base: Tensor,
    row_index: Tensor,
    rows: Tensor,
    *,
    width: int,
) -> None:
    """Write valid rows into a contiguous cache base.

    Example:
        >>> base = torch.zeros(4)
        >>> write_cache_rows(base, torch.tensor([1]), torch.ones(1, 2), width=2)
        >>> base.tolist()
        [0.0, 0.0, 1.0, 1.0]
    """
    view = _row_view(base, width)
    index = row_index.reshape(-1).to(torch.int64)
    if rows.shape != (index.shape[0], width):
        raise ValueError(
            f"The rows have shape {tuple(rows.shape)}. Use one row of {width} "
            "elements for each index."
        )
    if index.shape[0] == 0:
        return
    valid = (index >= 0) & (index < view.shape[0])
    if _can_use_row_kernel(base, width):
        _write_rows_kernel(
            base,
            view.shape[0],
            index,
            valid,
            rows,
            width,
        )
        return
    previous_row_zero = view[0].clone()
    safe = torch.where(valid, index, torch.zeros_like(index))
    current = view.index_select(0, safe)
    update = torch.where(valid.unsqueeze(-1), rows.to(view.dtype), current)
    view.index_copy_(0, safe, update)
    row_zero_writes = valid & (index == 0)
    positions = torch.arange(index.shape[0], device=index.device, dtype=torch.int64)
    last_row_zero = torch.where(
        row_zero_writes,
        positions,
        torch.zeros_like(positions),
    ).max()
    selected_row_zero = rows.to(view.dtype).index_select(
        0,
        last_row_zero.reshape(1),
    )[0]
    row_zero = torch.where(
        row_zero_writes.any(),
        selected_row_zero,
        previous_row_zero,
    )
    view[0].copy_(row_zero)


def read_cache_rows(
    base: Tensor,
    row_index: Tensor,
    *,
    width: int,
) -> Tensor:
    """Read valid rows and return zeros for invalid rows.

    Example:
        >>> base = torch.arange(4)
        >>> read_cache_rows(base, torch.tensor([1, -1]), width=2).tolist()
        [[2, 3], [0, 0]]
    """
    view = _row_view(base, width)
    index = row_index.reshape(-1).to(torch.int64)
    if index.shape[0] == 0:
        return view.new_zeros(0, width)
    valid = (index >= 0) & (index < view.shape[0])
    if _can_use_row_kernel(base, width):
        return _read_rows_kernel(
            base,
            view.shape[0],
            index,
            valid,
            width,
        )
    safe = torch.where(valid, index, torch.zeros_like(index))
    rows = view.index_select(0, safe)
    return torch.where(valid.unsqueeze(-1), rows, torch.zeros_like(rows))


def _validate_layout(
    bases: Sequence[Tensor],
    layout: MambaStateViewLayout,
) -> int:
    if not bases or not layout:
        raise ValueError(
            "The recurrent state cache or its layout is empty. "
            "Provide a cache base and a state layout."
        )
    num_pages = int(layout[0][2][0])
    if num_pages <= 0:
        raise ValueError(
            "The recurrent state cache has no null page. "
            "Allocate at least one cache page."
        )
    for base_index, dtype, shape, stride, offset in layout:
        if base_index < 0 or base_index >= len(bases):
            raise ValueError(
                f"The base index {base_index} is outside the cache base list. "
                "Use an index for an existing cache base."
            )
        if not shape or int(shape[0]) != num_pages:
            raise ValueError(
                "The recurrent states have different page counts. "
                "Use the same page count for each state."
            )
        _state_layout_in_base(bases[base_index], dtype, shape, stride, offset)
    return num_pages


def _state_layout_in_base(
    base: Tensor,
    dtype: torch.dtype,
    shape: Sequence[int],
    stride: Sequence[int],
    storage_offset: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Express one typed state page in elements of its carrier base."""
    payload = _contiguous_payload(shape, stride)
    state_itemsize = dtype.itemsize
    base_itemsize = base.element_size()
    if state_itemsize < base_itemsize or state_itemsize % base_itemsize:
        raise ValueError(
            f"The state data type {dtype} cannot use the carrier type "
            f"{base.dtype}. Use a compatible carrier data type."
        )
    scale = state_itemsize // base_itemsize
    base_shape = (int(shape[0]), payload * scale)
    base_stride = (int(stride[0]) * scale, 1)
    base_offset = int(storage_offset) * scale
    end = base_offset + (base_shape[0] - 1) * base_stride[0] + base_shape[1]
    if end > base.numel():
        raise ValueError(
            "The recurrent state layout exceeds its carrier base. "
            "Use a larger carrier base or a smaller layout."
        )
    return base_shape, base_stride, base_offset


def _contiguous_payload(
    shape: Sequence[int],
    stride: Sequence[int],
) -> int:
    if len(shape) != len(stride) or len(shape) < 2:
        raise ValueError(
            "The recurrent state has an invalid shape or stride. "
            "Provide a page dimension and a payload dimension."
        )
    expected = 1
    for size, step in zip(reversed(shape[1:]), reversed(stride[1:]), strict=True):
        if int(step) != expected:
            raise ValueError(
                "The state page payload is not contiguous. "
                "Use a contiguous page payload."
            )
        expected *= int(size)
    return expected


def _row_view(base: Tensor, width: int) -> Tensor:
    if not base.is_contiguous():
        raise ValueError("The cache base is not contiguous. Use the storage base.")
    if width < 1 or base.numel() % width:
        raise ValueError(
            f"The cache base has {base.numel()} elements, which the row width "
            f"{width} does not divide. Use a row width that divides the base."
        )
    return base.view(-1, width)


def _can_use_row_kernel(base: Tensor, width: int) -> bool:
    return can_run_kernel(base) and width * base.element_size() <= _MAX_ROW_BYTES


def _write_rows_kernel(
    base: Tensor,
    num_rows: int,
    index: Tensor,
    valid: Tensor,
    rows: Tensor,
    width: int,
) -> None:
    index_tile, pad = _tile_index(index, valid, num_rows)
    rows = rows.to(base.dtype)
    if pad:
        rows = torch.cat((rows, rows.new_zeros(pad, rows.shape[1])))
    shim = _row_shim_for(_WRITE_SHIMS, width, read=False)
    _call_row_kernel(
        shim,
        {0: 0},
        base,
        index_tile,
        rows.contiguous(),
        grid=_row_kernel_grid(index_tile.shape[0] // _PARTITIONS),
    )


def _read_rows_kernel(
    base: Tensor,
    num_rows: int,
    index: Tensor,
    valid: Tensor,
    width: int,
) -> Tensor:
    index_tile, _ = _tile_index(index, valid, num_rows)
    shim = _row_shim_for(_READ_SHIMS, width, read=True)
    rows = _call_row_kernel(
        shim,
        {},
        base,
        index_tile,
        grid=_row_kernel_grid(index_tile.shape[0] // _PARTITIONS),
    )
    return rows[: index.shape[0]]


def _call_row_kernel(
    shim,
    aliases: dict[int, int],
    *args: Tensor,
    grid: int | tuple[int, ...] | None = None,
):
    """Call one row kernel with its input and output alias map."""
    from libtorch_neuronx_lite.nki.nki_hop import (
        nki_kernel_wrapper,
        wrap_nki,
    )

    caller = wrap_nki(shim)[_row_kernel_grid() if grid is None else grid]
    return nki_kernel_wrapper(
        kernel_idx=caller.kernel_idx,
        grid=caller.grid,
        backend_config="",
        operand_output_aliases=aliases,
        args=args,
        arg_names=caller.kernel_arg_names,
        constant_args_key=-1,
    )


def _row_kernel_grid(tile_count: int | None = None) -> int | tuple[int, int]:
    """Return the row kernel grid for this process."""
    if envs.VLLM_NEURON_CPU_MODE:
        return 1
    lnc = get_logical_nc_config()
    if tile_count is None:
        return lnc
    return lnc, tile_count


def _tile_index(
    index: Tensor,
    valid: Tensor,
    num_rows: int,
) -> tuple[Tensor, int]:
    count = index.shape[0]
    tiles = -(-count // _PARTITIONS)
    pad = tiles * _PARTITIONS - count
    skip = num_rows + 1
    safe = torch.where(valid, index, torch.full_like(index, skip)).to(torch.int32)
    if pad:
        safe = torch.cat((safe, safe.new_full((pad,), skip)))
    return safe.reshape(tiles * _PARTITIONS, 1), pad


def _row_shim_for(
    shims: dict[int, object],
    width: int,
    *,
    read: bool,
):
    shim = shims.get(width)
    if shim is None:
        if torch.compiler.is_compiling():
            raise RuntimeError(
                f"The recurrent state row kernel has no trace shim for width "
                f"{width}. Call prewarm_mamba_state_access before graph capture."
            )
        shim = shims[width] = _build_row_shim(width, read=read)
    return shim


def _build_row_shim(width: int, read: bool = False):
    """Return a traced row kernel for one row width.

    The write must update the cache carrier in place, so the call uses
    ``nki_kernel_wrapper`` with an operand-output alias. ``wrap_nki`` does not
    take aliases. That call passes no constant arguments, so this shim binds
    ``width`` in a closure, one shim per width. The compile caches identify an
    NKI kernel by its function name, not by its source. The name therefore
    holds a digest of the kernel source, so an edited kernel never reuses a
    stale compiled graph.

    TODO: This works around gaps in ``wrap_nki`` in libtorch-neuronx-lite. Fix
    them there in a follow-up: take operand-output aliases and constant
    arguments, and key a compiled kernel by its source. Then call the row
    kernels through ``wrap_nki`` and remove this shim.
    """
    import nki

    from vllm_neuron._kernels.state_cache import recurrent_state_rows

    digest = hashlib.sha256(
        inspect.getsource(recurrent_state_rows).encode()
    ).hexdigest()
    if read:
        kernel = recurrent_state_rows.read_rows

        def trace(base, row_index):
            return kernel(base, row_index, width=width)

        name = f"read_recurrent_rows_w{width}_{digest[:12]}"
    else:
        kernel = recurrent_state_rows.write_rows_inplace

        def trace(base, row_index, rows):
            return kernel(base, row_index, rows, width=width)

        name = f"write_recurrent_rows_w{width}_{digest[:12]}"
    trace.__name__ = trace.__qualname__ = name
    return nki.jit(mode="trace")(trace)
