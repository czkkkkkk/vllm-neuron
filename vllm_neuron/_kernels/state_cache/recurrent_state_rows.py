# SPDX-License-Identifier: Apache-2.0
"""Read and write selected rows of a cache base with NKI kernels.

These model-agnostic kernels use a contiguous cache base with the shape
``[rows, width]``. A write returns the base as an alias. This operation avoids
a copy and a writeback of the complete cache allocation.

The kernels skip an invalid row. A read returns zero for that row. A write
does not change the base for that row.

Each kernel has a torch reference with the same signature for the NKL test
framework (``test/kernels/state_cache/``).
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nkilib.core.utils.allocator import sizeinbytes

PARTITIONS = 128
SBUF_BASE = 17 * 1024
SBUF_LIMIT = 224 * 1024 - 1024
MAX_ROW_BYTES = 8192


def kernel_assert(condition: bool, error_text: str) -> None:
    """Raise an NKI validation error when a condition is false.

    Example:
        >>> kernel_assert(True, "the condition must be true")
    """
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


@nki.jit
def write_rows_inplace(
    base: nl.NkiTensor,
    row_index: nl.NkiTensor,
    rows: nl.NkiTensor,
    width: int,
) -> nl.NkiTensor:
    """Write ``rows`` into ``base`` at ``row_index``.

    Example:
        >>> state = write_rows_inplace(  # doctest: +SKIP
        ...     state, row_index, rows, width=128
        ... )
    """
    if nl.program_ndim() != 0:
        if nl.program_id(axis=0) == 1:
            return base
    numel = 1
    for size in base.shape:
        numel *= size
    kernel_assert(numel % width == 0, "The width must divide the base.")
    num_rows, row_width = numel // width, width
    base_rows = base.reshape((num_rows, row_width))
    index_rows, one = row_index.shape
    kernel_assert(one == 1, "The row index must have one column.")
    kernel_assert(
        index_rows % PARTITIONS == 0,
        "The row index must contain complete tiles.",
    )
    tiles = index_rows // PARTITIONS
    kernel_assert(
        tuple(rows.shape) == (PARTITIONS * tiles, row_width),
        "The rows must have the shape [128 * tiles, width].",
    )
    kernel_assert(
        rows.dtype == base.dtype,
        "The rows and the base must have the same data type.",
    )
    kernel_assert(
        num_rows < (1 << 31),
        "The base contains too many rows. Use fewer than 2^31 rows.",
    )
    row_bytes = row_width * sizeinbytes(base.dtype)
    kernel_assert(
        row_bytes <= MAX_ROW_BYTES,
        "A row is larger than 8 KB. Use a smaller row width.",
    )
    index_bytes = 64
    kernel_assert(
        SBUF_BASE + index_bytes + MAX_ROW_BYTES <= SBUF_LIMIT,
        "The row buffers exceed the SBUF limit. Reduce the buffer size.",
    )

    index = nl.ndarray(
        (PARTITIONS, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name="state_row_index",
        address=(0, SBUF_BASE),
    )
    buffer = nl.ndarray(
        (PARTITIONS, row_width),
        dtype=base.dtype,
        buffer=nl.sbuf,
        name="state_write_rows",
        address=(0, SBUF_BASE + index_bytes),
    )
    if nl.program_ndim() < 2:
        for tile in range(tiles):
            _write_tile(base_rows, row_index, rows, index, buffer, tile, row_width)
    else:
        tile = nl.program_id(axis=1)
        _write_tile(base_rows, row_index, rows, index, buffer, tile, row_width)
    return base


def _write_tile(base_rows, row_index, rows, index, buffer, tile, row_width):
    """Write one tile of 128 rows from one NKI program."""
    nisa.dma_copy(
        dst=index,
        src=row_index[
            tile * PARTITIONS : (tile + 1) * PARTITIONS,
            0:1,
        ],
    )
    nisa.dma_copy(
        dst=buffer,
        src=rows[
            tile * PARTITIONS : (tile + 1) * PARTITIONS,
            0:row_width,
        ],
    )
    nisa.dma_copy(
        dst=base_rows.ap(
            pattern=[[row_width, PARTITIONS], [1, row_width]],
            offset=0,
            vector_offset=index,
            indirect_dim=0,
        ),
        src=buffer,
        oob_mode=nisa.oob_mode.skip,
    )


@nki.jit
def read_rows(
    base: nl.NkiTensor,
    row_index: nl.NkiTensor,
    width: int,
) -> nl.NkiTensor:
    """Read selected rows without an alias to the cache base.

    Example:
        >>> rows = read_rows(state, row_index, width=128)  # doctest: +SKIP
    """
    numel = 1
    for size in base.shape:
        numel *= size
    kernel_assert(numel % width == 0, "The width must divide the base.")
    num_rows, row_width = numel // width, width
    base_rows = base.reshape((num_rows, row_width))
    index_rows, one = row_index.shape
    kernel_assert(one == 1, "The row index must have one column.")
    kernel_assert(
        index_rows % PARTITIONS == 0,
        "The row index must contain complete tiles.",
    )
    tiles = index_rows // PARTITIONS
    kernel_assert(
        num_rows < (1 << 31),
        "The base contains too many rows. Use fewer than 2^31 rows.",
    )
    row_bytes = row_width * sizeinbytes(base.dtype)
    kernel_assert(
        row_bytes <= MAX_ROW_BYTES,
        "A row is larger than 8 KB. Use a smaller row width.",
    )
    index_bytes = 64
    kernel_assert(
        SBUF_BASE + index_bytes + 2 * MAX_ROW_BYTES <= SBUF_LIMIT,
        "The row buffers exceed the SBUF limit. Reduce the buffer size.",
    )

    output = nl.ndarray(
        (PARTITIONS * tiles, row_width),
        dtype=base.dtype,
        buffer=nl.shared_hbm,
        name="state_rows_output",
    )
    if nl.program_ndim() != 0:
        if nl.program_id(axis=0) == 1:
            return output
    index = nl.ndarray(
        (PARTITIONS, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name="state_read_index",
        address=(0, SBUF_BASE),
    )
    buffer = nl.ndarray(
        (PARTITIONS, row_width),
        dtype=base.dtype,
        buffer=nl.sbuf,
        name="state_read_rows",
        address=(0, SBUF_BASE + index_bytes),
    )
    if nl.program_ndim() < 2:
        for tile in range(tiles):
            _read_tile(
                base_rows,
                row_index,
                output,
                index,
                buffer,
                tile,
                row_width,
            )
    else:
        tile = nl.program_id(axis=1)
        _read_tile(
            base_rows,
            row_index,
            output,
            index,
            buffer,
            tile,
            row_width,
        )
    return output


def _read_tile(base_rows, row_index, output, index, buffer, tile, row_width):
    """Read one tile of 128 rows from one NKI program."""
    nisa.dma_copy(
        dst=index,
        src=row_index[
            tile * PARTITIONS : (tile + 1) * PARTITIONS,
            0:1,
        ],
    )
    nisa.memset(dst=buffer, value=0.0)
    nisa.dma_copy(
        dst=buffer,
        src=base_rows.ap(
            pattern=[[row_width, PARTITIONS], [1, row_width]],
            offset=0,
            vector_offset=index,
            indirect_dim=0,
        ),
        oob_mode=nisa.oob_mode.skip,
    )
    nisa.dma_copy(
        dst=output[
            tile * PARTITIONS : (tile + 1) * PARTITIONS,
            0:row_width,
        ],
        src=buffer,
    )


def _valid_rows(
    row_index: torch.Tensor, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the row index as int64 and the mask of rows inside the base."""
    index = row_index.reshape(-1).to(torch.int64)
    return index, (index >= 0) & (index < num_rows)


def read_rows_torch_ref(
    base: torch.Tensor,
    row_index: torch.Tensor,
    width: int,
) -> dict[str, torch.Tensor]:
    """Return the selected rows of ``base``, with zeros for invalid rows.

    Args:
        base: Cache base with ``width`` dividing its element count.
        row_index: ``[128 * tiles, 1]`` row index. An index outside the base
            is invalid.
        width: Elements in one row.

    Returns:
        ``{"output": rows}`` with shape ``[128 * tiles, width]``.

    Example:
        >>> read_rows_torch_ref(torch.arange(8.0), torch.tensor([[1], [9]]), 4)
        {'output': tensor([[4., 5., 6., 7.],
                [0., 0., 0., 0.]])}
    """
    base_rows = base.reshape(-1, width)
    index, valid = _valid_rows(row_index, base_rows.shape[0])
    output = torch.zeros((index.shape[0], width), dtype=base.dtype)
    output[valid] = base_rows[index[valid]]
    return {"output": output}


def write_rows_inplace_torch_ref(
    base: torch.Tensor,
    row_index: torch.Tensor,
    rows: torch.Tensor,
    width: int,
) -> dict[str, torch.Tensor]:
    """Return ``base`` with ``rows`` written at the valid ``row_index`` rows.

    Args:
        base: Cache base with ``width`` dividing its element count.
        row_index: ``[128 * tiles, 1]`` row index. An index outside the base
            is invalid, and its row is not written.
        rows: ``[128 * tiles, width]`` rows to write.
        width: Elements in one row.

    Returns:
        ``{"base": updated}`` with the shape of ``base``.

    Example:
        >>> write_rows_inplace_torch_ref(
        ...     torch.zeros(8), torch.tensor([[1], [9]]), torch.ones(2, 4), 4
        ... )["base"].tolist()
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    """
    updated = base.clone()
    base_rows = updated.reshape(-1, width)
    index, valid = _valid_rows(row_index, base_rows.shape[0])
    base_rows[index[valid]] = rows[valid]
    return {"base": updated}
