# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    MambaAttentionBackendEnum,
    MambaSpec,
)


class KeyOnlyAttentionSpec(MLAAttentionSpec):
    """Key-only KV cache spec that will not merge caches of different widths.

    ``MLAAttentionSpec`` budgets one vector per token rather than a K/V pair,
    which is what a key-only cache needs. Its ``merge`` asserts on the
    quantization fields but takes ``head_size``/``num_kv_heads``/``dtype`` from
    the first spec, and vLLM decides two layers share a KV cache group by
    calling ``merge`` and checking whether it raises
    (``kv_cache_utils.is_kv_cache_spec_uniform``). A 576-wide MLA cache and a
    128-wide indexer cache therefore merge silently into one 576-wide group,
    and every layer in that group is budgeted AND shaped at 576 -- the indexer
    both wastes 448 columns and rejects its own 128-wide write.

    Asserting on the width keeps differing caches out of one group, so vLLM
    falls through to ``UniformTypeKVCacheSpecs``, which budgets each layer at
    its own page size and exposes per-layer specs to the worker.
    """

    @classmethod
    def merge(cls, specs):
        widths = {(s.num_kv_heads, s.head_size, s.dtype) for s in specs}
        assert len(widths) == 1, (
            "KV caches of different widths cannot share a cache group: "
            f"{sorted(str(w) for w in widths)}"
        )
        return super().merge(specs)


@dataclass
class LayerSpec:
    """Define the KV cache specification for one attention layer.

    Example:
        >>> LayerSpec(
        ...     name="layers.0.attn",
        ...     num_kv_heads=8,
        ...     head_size=128,
        ...     dtype=torch.bfloat16,
        ... )
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None
    # One vector per token instead of a K/V pair, budgeted and allocated as a
    # single cache. Set by MLA layers, which store one combined
    # [rope | compressed_kv] entry, and by key-only caches such as the DeepSeek
    # V3.2 sparse-attention indexer. bind_kv_cache receives a 1-element list.
    key_only: bool = False


@dataclass
class RecurrentLayerSpec:
    """Define the recurrent cache specification for one model layer.

    Example:
        >>> RecurrentLayerSpec(
        ...     name="layers.0.linear_attn",
        ...     state_shapes=((3, 16),),
        ...     state_dtypes=(torch.bfloat16,),
        ...     state_names=("conv",),
        ...     state_backend_type=MambaAttentionBackendEnum.MAMBA2,
        ... )
    """

    name: str
    state_shapes: tuple[tuple[int, ...], ...]
    state_dtypes: tuple[torch.dtype, ...]
    state_names: tuple[str, ...]
    state_backend_type: MambaAttentionBackendEnum
    state_num_heads: int = 1
    state_tokens_per_state: int = -1
    state_tp_replicated: bool = False
    # Align-mode APC can need one more state snapshot inside a prefill. The
    # scheduler reserves this many transient pages for it. The model declares
    # the token alignment at which it can export the snapshot cheaply.
    state_num_prefill_checkpoint_blocks: int = 0
    state_prefill_checkpoint_alignment: int | None = None

    def __post_init__(self) -> None:
        """Validate the recurrent cache declaration."""
        if not self.state_shapes:
            raise ValueError(
                f"{self.name}: The recurrent state has no tensors. "
                "Define at least one state tensor."
            )
        if len(self.state_shapes) != len(self.state_dtypes):
            raise ValueError(
                f"{self.name}: The state shape count differs from the dtype count. "
                "Define one dtype for each state shape."
            )
        if self.state_names is None:
            raise ValueError(
                f"{self.name}: The recurrent state has no state_names value. "
                "Set one name for each state tensor."
            )
        if len(self.state_shapes) != len(self.state_names):
            raise ValueError(
                f"{self.name}: The state shape count differs from the name count. "
                "Define one name for each state shape."
            )
        if len(set(self.state_names)) != len(self.state_names) or any(
            not name for name in self.state_names
        ):
            raise ValueError(
                f"{self.name}: A state name is empty or is not unique. "
                "Use a unique nonempty name for each state tensor."
            )
        if any(
            not shape or any(dimension <= 0 for dimension in shape)
            for shape in self.state_shapes
        ):
            raise ValueError(
                f"{self.name}: A state shape is empty or has a nonpositive dimension. "
                "Use positive dimensions in each state shape."
            )
        if self.state_backend_type is None:
            raise ValueError(
                f"{self.name}: The recurrent state has no backend type. "
                "Set state_backend_type."
            )
        if self.state_num_heads <= 0:
            raise ValueError(
                f"{self.name}: state_num_heads is not positive. "
                "Set state_num_heads to a positive value."
            )
        if self.state_tokens_per_state == 0 or self.state_tokens_per_state < -1:
            raise ValueError(
                f"{self.name}: state_tokens_per_state is invalid. "
                "Set it to -1 or a positive value."
            )
        if self.state_num_prefill_checkpoint_blocks < 0:
            raise ValueError(
                f"{self.name}: state_num_prefill_checkpoint_blocks is negative. "
                "Set it to zero or a positive value."
            )
        if (self.state_num_prefill_checkpoint_blocks == 0) != (
            self.state_prefill_checkpoint_alignment is None
        ):
            raise ValueError(
                f"{self.name}: The prefill checkpoint count and alignment are "
                "incomplete. Set both values together."
            )
        if (
            self.state_prefill_checkpoint_alignment is not None
            and self.state_prefill_checkpoint_alignment <= 0
        ):
            raise ValueError(
                f"{self.name}: state_prefill_checkpoint_alignment is not positive. "
                "Set it to a positive value."
            )


@dataclass
class KVSpec:
    """Define the cache needs for all model layers.

    Example:
        >>> KVSpec(
        ...     layers=[
        ...         LayerSpec("layers.0.attn", 8, 128, torch.bfloat16),
        ...     ]
        ... )
    """

    layers: list[LayerSpec]


MambaStateViewLayout = tuple[
    tuple[int, torch.dtype, tuple[int, ...], tuple[int, ...], int],
    ...,
]


@dataclass(frozen=True)
class MambaStateTransition:
    """Describe one recurrent-state cache transition.

    Invalid reads and writes point to the reserved null page, so no page ID is
    negative. The resolvers do not know the pool size, so callers bound the
    upper page ID, for example with ``mamba_state_page_ids``. The validity
    tensors describe the logical state transition without requiring each model
    layer to mask the indices again.

    A read page can also be a write page of the same row. Consumers must load
    the complete read state before they store any checkpoint.
    """

    read_page_ids: torch.Tensor
    write_page_ids: torch.Tensor
    read_state_valid: torch.Tensor
    write_valid_mask: torch.Tensor
    prefill_checkpoint_offsets: torch.Tensor | None = None

    def as_attention_metadata(self) -> dict[str, torch.Tensor]:
        """Return the tensors that a model adapter reads.

        Returns:
            A dictionary that maps recurrent metadata names to transition
            tensors.

        Example:
            >>> page = torch.tensor([[5]], dtype=torch.int32)
            >>> transition = MambaStateTransition(
            ...     read_page_ids=page,
            ...     write_page_ids=page,
            ...     read_state_valid=torch.tensor([True]),
            ...     write_valid_mask=torch.tensor([[True]]),
            ... )
            >>> transition.as_attention_metadata()[
            ...     "recurrent_state_read_page_ids"
            ... ].tolist()
            [[5]]
        """
        metadata = {
            "recurrent_state_read_page_ids": self.read_page_ids,
            "recurrent_state_write_page_ids": self.write_page_ids,
            "recurrent_state_read_valid": self.read_state_valid,
            "recurrent_state_write_valid_mask": self.write_valid_mask,
        }
        if self.prefill_checkpoint_offsets is not None:
            metadata["recurrent_state_prefill_checkpoint_offsets"] = (
                self.prefill_checkpoint_offsets
            )
        return metadata


def _validate_recurrent_block_table(
    block_table: torch.Tensor,
) -> None:
    """Validate the common recurrent block-table shape."""
    if block_table.ndim != 2:
        raise ValueError(
            "The recurrent block table is not two-dimensional. "
            "Pass a table with shape [requests, blocks]."
        )


def _validate_num_speculative_blocks(num_speculative_blocks: int) -> None:
    """Validate the number of speculative recurrent pages."""
    if num_speculative_blocks < 0:
        raise ValueError(
            "The speculative recurrent block count is negative. "
            "Set it to zero or a positive value."
        )


def _validate_prefill_checkpoints(
    checkpoint_columns: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    *,
    computed: torch.Tensor,
    scheduled: torch.Tensor,
    write_columns: torch.Tensor,
    block_size: int,
) -> None:
    """Check that each internal prefill checkpoint has its own page.

    The checkpoint must land after the page that holds the initial state and
    before the running page. Otherwise it overwrites a page that the step also
    reads, which can be a shared prefix page, or a page that it also writes.
    """
    has_checkpoint = checkpoint_columns >= 0
    if not bool(has_checkpoint.any()):
        return
    # Floor division keeps -1 for a request that starts from token zero.
    initial_columns = torch.div(computed - 1, block_size, rounding_mode="floor")
    offsets = checkpoint_offsets.to(torch.int64)
    misplaced = has_checkpoint & (
        (checkpoint_columns <= initial_columns)
        | (checkpoint_columns >= write_columns)
        | (offsets <= 0)
        | (offsets >= scheduled)
    )
    if bool(misplaced.any()):
        rows = torch.nonzero(misplaced).reshape(-1).tolist()
        raise ValueError(
            f"The prefill checkpoint for rows {rows} does not have its own page. "
            "Place each checkpoint after the initial-state page, before the "
            "running page, and inside the scheduled tokens."
        )


def _apply_output_checkpoint_mask(
    write_page_ids: torch.Tensor,
    write_valid_mask: torch.Tensor,
    output_checkpoint_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the optional output mask and keep invalid page IDs safe."""
    if output_checkpoint_valid is None:
        return write_page_ids, write_valid_mask

    checkpoint_valid = output_checkpoint_valid.to(
        device=write_page_ids.device,
        dtype=torch.bool,
    )
    if checkpoint_valid.shape != write_valid_mask.shape:
        raise ValueError(
            "The output checkpoint mask shape differs from the page ID shape. "
            "Pass one mask value for each output page ID."
        )
    write_valid_mask = write_valid_mask & checkpoint_valid
    write_page_ids = torch.where(
        write_valid_mask,
        write_page_ids,
        torch.zeros_like(write_page_ids),
    )
    return write_page_ids, write_valid_mask


def resolve_aligned_mamba_state_transition(
    block_table: torch.Tensor,
    *,
    num_speculative_blocks: int,
    num_computed_tokens: torch.Tensor,
    num_scheduled_tokens: torch.Tensor,
    block_size: int,
    state_cold: torch.Tensor | None = None,
    num_output_checkpoints: int = 1,
    output_checkpoint_valid: torch.Tensor | None = None,
    prefill_checkpoint_columns: torch.Tensor | None = None,
    prefill_checkpoint_offsets: torch.Tensor | None = None,
) -> MambaStateTransition:
    """Resolve a position-aligned recurrent-state transition on the host.

    Args:
        block_table: Physical page IDs with shape ``[requests, blocks]``.
        num_speculative_blocks: Number of speculative checkpoint pages.
        num_computed_tokens: Token count before this step for each request.
        num_scheduled_tokens: Token count scheduled for each request.
        block_size: Number of tokens represented by one state page.
        state_cold: Optional cold-state mask for each request.
        num_output_checkpoints: Number of state checkpoints that the model
            writes.
        output_checkpoint_valid: Optional validity mask for output checkpoints.
        prefill_checkpoint_columns: Optional internal prefill checkpoint
            columns.
        prefill_checkpoint_offsets: Token offset for each internal prefill
            checkpoint.

    Returns:
        The resolved aligned recurrent-state transition.

    Example:
        >>> transition = resolve_aligned_mamba_state_transition(
        ...     torch.tensor([[5]], dtype=torch.int32),
        ...     num_speculative_blocks=0,
        ...     num_computed_tokens=torch.tensor([0]),
        ...     num_scheduled_tokens=torch.tensor([1]),
        ...     block_size=1,
        ... )
        >>> transition.write_page_ids.tolist()
        [[5]]
    """
    _validate_recurrent_block_table(block_table)
    _validate_num_speculative_blocks(num_speculative_blocks)
    if num_speculative_blocks:
        raise NotImplementedError(
            "Aligned recurrent state does not support speculative decoding. "
            "Set num_speculative_blocks to zero."
        )
    if block_table.shape[1] == 0:
        raise ValueError(
            "The aligned recurrent block table has no columns. "
            "Pass at least one block column."
        )
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(
            "The Mamba block size is missing or is not positive. "
            "Pass a positive block size."
        )

    if prefill_checkpoint_columns is None:
        checkpoint_columns = torch.empty(
            (block_table.shape[0], 0),
            dtype=torch.int64,
            device=block_table.device,
        )
    else:
        checkpoint_columns = prefill_checkpoint_columns.to(
            device=block_table.device,
            dtype=torch.int64,
        )
        if checkpoint_columns.ndim == 1:
            checkpoint_columns = checkpoint_columns.unsqueeze(1)
        if (
            checkpoint_columns.ndim != 2
            or checkpoint_columns.shape[0] != block_table.shape[0]
        ):
            raise ValueError(
                "The prefill checkpoint columns have an invalid shape. "
                "Pass a tensor with shape [requests, checkpoints]."
            )
        if bool((checkpoint_columns < -1).any()):
            raise ValueError(
                "A prefill checkpoint column is below minus one. "
                "Use minus one or a nonnegative column."
            )
    if num_output_checkpoints != checkpoint_columns.shape[1] + 1:
        raise ValueError(
            "The aligned recurrent output count is invalid. Include each "
            "prefill checkpoint and one running checkpoint."
        )

    if prefill_checkpoint_offsets is None:
        checkpoint_offsets = None
        if checkpoint_columns.shape[1]:
            raise ValueError(
                "The aligned prefill checkpoints have no token offsets. "
                "Pass one token offset for each request."
            )
    else:
        checkpoint_offsets = prefill_checkpoint_offsets.reshape(-1).to(
            device=block_table.device,
            dtype=torch.int32,
        )
        if checkpoint_offsets.shape != block_table.shape[:1]:
            raise ValueError(
                "The prefill checkpoint offset count differs from the request "
                "count. Pass one offset for each request."
            )
        if checkpoint_columns.shape[1] == 0:
            raise ValueError(
                "The aligned prefill checkpoint offsets have no checkpoint "
                "columns. Pass one checkpoint column for each request."
            )
        if checkpoint_columns.shape[1] != 1:
            raise ValueError(
                "The recurrent metadata has more than one internal prefill "
                "checkpoint per request. Pass one internal checkpoint."
            )

    def _request_counts(value: torch.Tensor, name: str) -> torch.Tensor:
        counts = value.reshape(-1).to(
            device=block_table.device,
            dtype=torch.int64,
        )
        if counts.shape != block_table.shape[:1]:
            raise ValueError(
                f"The {name} count differs from the request count. "
                "Pass one value per request."
            )
        if bool((counts < 0).any()):
            raise ValueError(
                f"{name} contains a negative value. "
                "Pass zero or a positive value for each request."
            )
        return counts

    computed = _request_counts(num_computed_tokens, "num_computed_tokens")
    scheduled = _request_counts(num_scheduled_tokens, "num_scheduled_tokens")
    active = scheduled > 0
    read_columns = torch.div(
        torch.clamp(computed - 1, min=0),
        block_size,
        rounding_mode="floor",
    )
    write_columns = torch.div(
        torch.clamp(computed + scheduled - 1, min=0),
        block_size,
        rounding_mode="floor",
    )
    read_columns = torch.where(
        active,
        read_columns,
        torch.full_like(read_columns, -1),
    )
    write_columns = torch.where(
        active,
        write_columns,
        torch.full_like(write_columns, -1),
    )
    output_columns = torch.cat(
        (checkpoint_columns, write_columns.unsqueeze(1)),
        dim=1,
    )
    output_columns = torch.where(
        active.unsqueeze(1),
        output_columns,
        torch.full_like(output_columns, -1),
    )
    if checkpoint_offsets is not None:
        _validate_prefill_checkpoints(
            output_columns[:, 0],
            checkpoint_offsets,
            computed=computed,
            scheduled=scheduled,
            write_columns=write_columns,
            block_size=block_size,
        )

    valid_columns = torch.cat((read_columns, output_columns.reshape(-1)))
    max_column = int(valid_columns.max().item()) if valid_columns.numel() else -1
    if max_column >= block_table.shape[1]:
        raise ValueError(
            f"Aligned recurrent state selected column {max_column}, but the "
            f"block table has only {block_table.shape[1]} columns. "
            "Increase the table width or select a valid column."
        )

    page_ids = block_table.to(torch.int32)
    page_valid = page_ids > 0
    safe_page_ids = torch.where(page_valid, page_ids, torch.zeros_like(page_ids))
    read_column_valid = read_columns >= 0
    read_indices = read_columns.clamp(min=0).unsqueeze(1)
    read_page_ids = torch.gather(safe_page_ids, 1, read_indices)
    read_page_valid = (
        read_column_valid
        & torch.gather(
            page_valid,
            1,
            read_indices,
        )[:, 0]
    )
    read_page_ids = torch.where(
        read_page_valid.unsqueeze(1),
        read_page_ids,
        torch.zeros_like(read_page_ids),
    )

    output_column_valid = output_columns >= 0
    safe_output_columns = output_columns.clamp(min=0)
    write_page_ids = torch.gather(
        safe_page_ids,
        1,
        safe_output_columns,
    )
    write_valid_mask = output_column_valid & torch.gather(
        page_valid,
        1,
        safe_output_columns,
    )
    write_page_ids = torch.where(
        write_valid_mask,
        write_page_ids,
        torch.zeros_like(write_page_ids),
    )

    # A request with no computed tokens has no state to read, even when the
    # caller's cold mask says otherwise.
    cold = computed == 0
    if state_cold is not None:
        caller_cold = state_cold.reshape(-1).to(
            device=block_table.device,
            dtype=torch.bool,
        )
        if caller_cold.shape != read_page_valid.shape:
            raise ValueError(
                "The cold-state mask count differs from the request count. "
                "Pass one value for each request."
            )
        cold = cold | caller_cold
    read_state_valid = read_page_valid & ~cold
    write_page_ids, write_valid_mask = _apply_output_checkpoint_mask(
        write_page_ids,
        write_valid_mask,
        output_checkpoint_valid,
    )
    if checkpoint_offsets is not None:
        # A row without a valid checkpoint page exports no checkpoint.
        checkpoint_offsets = torch.where(
            write_valid_mask[:, 0],
            checkpoint_offsets,
            torch.zeros_like(checkpoint_offsets),
        )

    return MambaStateTransition(
        read_page_ids=read_page_ids,
        write_page_ids=write_page_ids,
        read_state_valid=read_state_valid,
        write_valid_mask=write_valid_mask,
        prefill_checkpoint_offsets=checkpoint_offsets,
    )


def resolve_fixed_mamba_state_transition(
    block_table: torch.Tensor,
    *,
    num_speculative_blocks: int,
    state_cold: torch.Tensor | None = None,
    num_output_checkpoints: int = 1,
    output_checkpoint_valid: torch.Tensor | None = None,
) -> MambaStateTransition:
    """Resolve a fixed-page recurrent-state transition.

    Args:
        block_table: Physical page IDs with shape ``[requests, blocks]``.
        num_speculative_blocks: Number of speculative checkpoint pages.
        state_cold: Optional cold-state mask for each request.
        num_output_checkpoints: Number of state checkpoints that the model
            writes.
        output_checkpoint_valid: Optional validity mask for output checkpoints.

    Returns:
        The resolved fixed-page recurrent-state transition.

    Example:
        >>> transition = resolve_fixed_mamba_state_transition(
        ...     torch.tensor([[5]], dtype=torch.int32),
        ...     num_speculative_blocks=0,
        ... )
        >>> transition.read_page_ids.tolist()
        [[5]]
    """
    _validate_recurrent_block_table(block_table)
    _validate_num_speculative_blocks(num_speculative_blocks)
    expected_width = 1 + num_speculative_blocks
    if block_table.shape[1] != expected_width:
        raise ValueError(
            f"The recurrent block table has {block_table.shape[1]} columns. "
            f"Expected {expected_width} columns. Pass the expected table width."
        )
    if num_output_checkpoints <= 0 or num_output_checkpoints > expected_width:
        raise ValueError(
            f"The output checkpoint count is outside 1 to {expected_width}. "
            "Pass a count in this range."
        )

    page_ids = block_table.to(torch.int32)
    page_valid = page_ids > 0
    safe_page_ids = torch.where(page_valid, page_ids, torch.zeros_like(page_ids))
    read_page_ids = safe_page_ids[:, :1]
    read_page_valid = page_valid[:, 0]

    if state_cold is None:
        read_state_valid = read_page_valid
    else:
        cold = state_cold.reshape(-1).to(
            device=block_table.device,
            dtype=torch.bool,
        )
        if cold.shape != read_page_valid.shape:
            raise ValueError(
                "The cold-state mask count differs from the request count. "
                "Pass one value for each request."
            )
        read_state_valid = read_page_valid & ~cold

    if num_output_checkpoints == expected_width:
        # Keep the full-width view and its established alias topology.
        write_page_ids = safe_page_ids[:, :num_output_checkpoints]
        write_valid_mask = page_valid[:, :num_output_checkpoints]
    else:
        # A prefix slice of a wider table has a row stride larger than its
        # logical width. The Neuron runtime cannot pass that non-contiguous
        # tensor to the compiled model when the batch size is greater than one.
        output_columns = torch.arange(
            num_output_checkpoints,
            dtype=torch.long,
            device=block_table.device,
        )
        write_page_ids = torch.index_select(
            safe_page_ids,
            1,
            output_columns,
        )
        write_valid_mask = torch.index_select(
            page_valid,
            1,
            output_columns,
        )
    write_page_ids, write_valid_mask = _apply_output_checkpoint_mask(
        write_page_ids,
        write_valid_mask,
        output_checkpoint_valid,
    )

    if expected_width == 1:
        # Preserve the width-one input identity that the existing compiled
        # graphs use. The source and output are the same physical page, so a
        # warm read cannot independently mask that page as an invalid output.
        masked_warm_read = read_state_valid & ~write_valid_mask[:, 0]
        if bool(masked_warm_read.any()):
            raise ValueError(
                "A width-one recurrent transition masks the output page of a "
                "valid read. Keep the output valid or mark the recurrent state cold."
            )
        read_page_ids = write_page_ids

    # Keep the physical index safe even when the logical row is invalid.
    # Consumers use read_state_valid/write_valid_mask to distinguish those
    # rows, so a negative sentinel would only create an out-of-bounds hazard.
    return MambaStateTransition(
        read_page_ids=read_page_ids,
        write_page_ids=write_page_ids,
        read_state_valid=read_state_valid,
        write_valid_mask=write_valid_mask,
    )


def carve_mamba_state_views(
    raw_tensor: torch.Tensor,
    spec: MambaSpec,
) -> list[torch.Tensor]:
    """Create page-major state views from a raw Mamba cache tensor.

    Each output has the shape ``[num_pages, *state_shape]``. The outer stride
    skips one complete physical page.

    Example:
        >>> spec = MambaSpec(
        ...     shapes=((4, 3), (2, 4, 4)),
        ...     dtypes=(torch.bfloat16, torch.float32),
        ...     block_size=1,
        ... )
        >>> raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.int8)
        >>> [view.shape for view in carve_mamba_state_views(raw, spec)]
        [torch.Size([2, 4, 3]), torch.Size([2, 2, 4, 4])]
    """
    if raw_tensor.element_size() != 1:
        raise ValueError(
            "The raw Mamba cache dtype uses more than one byte. "
            "Use a one-byte storage dtype."
        )
    if not raw_tensor.is_contiguous():
        raise ValueError(
            "The raw Mamba cache is not contiguous. Pass a contiguous tensor."
        )
    if raw_tensor.numel() % spec.page_size_bytes:
        raise ValueError(
            "The raw Mamba cache size contains an incomplete page. "
            "Allocate a whole number of pages."
        )

    num_pages = raw_tensor.numel() // spec.page_size_bytes
    views: list[torch.Tensor] = []
    typed_views: dict[torch.dtype, torch.Tensor] = {}
    storage_offset_bytes = 0

    for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True):
        dtype_size = torch.empty((), dtype=dtype).element_size()
        if spec.page_size_bytes % dtype_size:
            raise ValueError(
                "The Mamba page size does not align with a state dtype. "
                "Use a page size that aligns with each dtype."
            )
        if storage_offset_bytes % dtype_size:
            raise ValueError(
                "The Mamba state offset does not align with its dtype. "
                "Use an aligned state layout."
            )

        target_shape = (num_pages, *shape)
        contiguous_stride = torch.empty(target_shape, device="meta").stride()
        page_stride = spec.page_size_bytes // dtype_size
        target_stride = (page_stride, *contiguous_stride[1:])

        typed_view = typed_views.get(dtype)
        if typed_view is None:
            typed_view = raw_tensor.view(dtype)
            typed_views[dtype] = typed_view

        views.append(
            torch.as_strided(
                typed_view,
                size=target_shape,
                stride=target_stride,
                storage_offset=storage_offset_bytes // dtype_size,
            )
        )
        storage_offset_bytes += contiguous_stride[0] * dtype_size

    if storage_offset_bytes > spec.page_size_bytes:
        raise ValueError(
            "The Mamba states are larger than one physical page. "
            "Increase the page size or reduce the state sizes."
        )
    return views


def prepare_mamba_state_cache(
    state_views: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], MambaStateViewLayout]:
    """Replace state views with one carrier base per physical allocation.

    Mixed-dtype states in one upstream Mamba page share the same raw storage.
    Passing one typed base per dtype makes AOTAutograd reject in-place cache
    updates because the inputs are different-dtype views of one allocation.
    Keep one contiguous carrier base instead, choosing the smallest element
    dtype already present in each allocation. The static layout records every
    state's declared dtype so the model can restore or bit-pack its view.

    Example:
        >>> base = torch.zeros(16, dtype=torch.bfloat16)
        >>> view = torch.as_strided(base, (2, 4), (8, 1))
        >>> bases, layout = prepare_mamba_state_cache([view])
        >>> restore_mamba_state_views(bases, layout)[0].stride()
        (8, 1)
    """
    if not state_views:
        raise ValueError(
            "The Mamba state cache has no views. Pass one or more state views."
        )

    records: list[
        tuple[
            tuple[torch.device, int],
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    carriers: dict[tuple[torch.device, int], torch.Tensor] = {}
    storage_order: list[tuple[torch.device, int]] = []

    for view in state_views:
        base = view
        while base._base is not None and base._base.dtype == view.dtype:
            base = base._base
        if base.dtype != view.dtype or not base.is_contiguous():
            raise ValueError(
                "The Mamba state view has no contiguous typed base. "
                "Create the view from a contiguous tensor with the same dtype."
            )
        storage = base.untyped_storage()
        if (
            base.storage_offset() != 0
            or base.numel() * base.element_size() != storage.nbytes()
        ):
            raise ValueError(
                "The Mamba state typed base does not cover its complete allocation. "
                "Pass the complete storage base."
            )

        storage_key = (base.device, storage._cdata)
        carrier = carriers.get(storage_key)
        if carrier is None:
            carriers[storage_key] = base
            storage_order.append(storage_key)
        elif base.element_size() < carrier.element_size():
            carriers[storage_key] = base
        records.append((storage_key, base, view))

    bases = [carriers[key] for key in storage_order]
    base_indices = {key: index for index, key in enumerate(storage_order)}
    layout: list[tuple[int, torch.dtype, tuple[int, ...], tuple[int, ...], int]] = []
    for storage_key, _typed_base, view in records:
        carrier = carriers[storage_key]
        if view.element_size() % carrier.element_size():
            raise ValueError(
                "The Mamba state dtype does not align with its storage carrier dtype. "
                "Use compatible element sizes."
            )
        layout.append(
            (
                base_indices[storage_key],
                view.dtype,
                tuple(view.shape),
                tuple(view.stride()),
                view.storage_offset(),
            )
        )

    return bases, tuple(layout)


def restore_mamba_state_views(
    bases: Sequence[torch.Tensor],
    layout: MambaStateViewLayout,
) -> list[torch.Tensor]:
    """Restore page views from contiguous state-cache bases.

    Example:
        >>> base = torch.zeros(16, dtype=torch.bfloat16)
        >>> layout = ((0, torch.bfloat16, (2, 4), (8, 1), 0),)
        >>> restore_mamba_state_views([base], layout)[0].shape
        torch.Size([2, 4])
    """
    views: list[torch.Tensor] = []
    for base_index, dtype, shape, stride, storage_offset in layout:
        if base_index < 0 or base_index >= len(bases):
            raise ValueError(
                "The Mamba state layout has an invalid base index. "
                "Use an index for an available storage base."
            )
        base = bases[base_index]
        typed_base = base if base.dtype == dtype else base.view(dtype)
        views.append(
            torch.as_strided(
                typed_base,
                size=shape,
                stride=stride,
                storage_offset=storage_offset,
            )
        )
    return views


def mamba_state_page_ids(
    block_table: torch.Tensor,
    num_pages: int,
) -> torch.Tensor:
    """Return bounds-checked recurrent-state pages.

    Block 0 is the null page. Padding and out-of-range IDs also become page 0.
    The function preserves the input shape, including speculative columns.

    Args:
        block_table: Per-layer scheduler block table.
        num_pages: Number of physical pages in the recurrent-state pool.

    Returns:
        An int32 tensor with the input shape. Valid scheduler page IDs pass
        through. Invalid IDs become the protected null page 0.

    Example:
        >>> table = torch.tensor([[3, 4], [-1, 2], [0, 7]])
        >>> mamba_state_page_ids(table, num_pages=7).tolist()
        [[3, 4], [0, 2], [0, 0]]
    """
    if block_table.ndim != 2 or block_table.shape[1] < 1:
        raise ValueError(
            "The recurrent-state block table has an invalid shape. "
            "Pass a table with shape [requests, blocks]."
        )
    if num_pages <= 0:
        raise ValueError(
            "The recurrent-state page pool must contain a null page, but it is empty. "
            "Allocate page zero as the null page."
        )

    page_ids = block_table.to(torch.int32)
    valid = (page_ids > 0) & (page_ids < num_pages)
    return torch.where(valid, page_ids, torch.zeros_like(page_ids))


# ---------------------------------------------------------------------------
# Paged-KV slot arithmetic
# ---------------------------------------------------------------------------
# Write-side padding target: the code redirects an invalid slot to block 0. The
# scheduler reserves block 0 as the null block for this purpose. This mirrors
# ``NULL_BLOCK_ID`` in ``neuron_model_runner``, whose context-parallel path
# already does ``np.where(oob_mask, NULL_BLOCK_ID, slot_mapping)``.
_NULL_BLOCK_ID = 0


def paged_slot_indices(
    slot_mapping: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive bounds-checked ``(block_indices, position_indices)`` for a KV write.

    The scheduler marks an inactive write slot with ``PAD_SLOT_ID`` (-1). A
    negative entry is not safe to use as a cache coordinate. ``-1 // block_size``
    floor-divides to ``-1``, not to ``0``. Eager torch wraps that ``-1`` to the
    LAST block and corrupts the KV of a live sequence. On Neuron the same
    ``index_put_`` becomes an HLO ``scatter`` that runs as an indirect vector-DGE
    copy, so ``-1`` becomes an out-of-range HBM address. The runtime then raises
    "scatter/gather (indirect memory copy via vector DGE) out-of-bound access",
    which wedges the NEFF execution queue and stops the engine core.

    This function redirects an entry outside ``[0, num_blocks * block_size)`` to
    the null block before it splits the slot into a block and a position. The
    write for that token still happens, and the null block discards it. A
    redirect keeps the returned shapes static, so the enclosing ``index_put_``
    stays one traceable operation.

    Call this function instead of a direct ``//`` and ``%`` on a slot mapping.
    It clamps before it derives both coordinates. A clamp after the ``//`` and
    the ``%`` leaves ``position_indices`` at ``block_size - 1`` for a pad entry.

    Args:
        slot_mapping: ``[num_tokens]`` flat slot ids, possibly padded with -1.
        block_size: Tokens per KV-cache block.
        num_blocks: Block count of the target cache (its dim-0 extent).

    Returns:
        ``(block_indices, position_indices)``. Both have the shape and the dtype
        of ``slot_mapping``. Every value is in range for ``num_blocks`` blocks.

    Example:
        >>> import torch
        >>> slots = torch.tensor([0, 5, -1, 9])  # -1 is PAD_SLOT_ID; 9 overflows
        >>> blocks, positions = paged_slot_indices(slots, 4, 2)
        >>> blocks
        tensor([0, 1, 0, 0])
        >>> positions
        tensor([0, 1, 0, 0])
    """
    max_slot = num_blocks * block_size
    safe_slots = torch.where(
        (slot_mapping < 0) | (slot_mapping >= max_slot),
        torch.full_like(slot_mapping, _NULL_BLOCK_ID),
        slot_mapping,
    )
    return safe_slots // block_size, safe_slots % block_size


def build_per_head_slot_mapping(
    slot_mapping: torch.Tensor,
    block_size: int,
    kv_heads: int,
    batch_size: int,
) -> torch.Tensor:
    """Build the per-head ``kv_cache_update_idx`` of the in-kernel KV cache update.

    ``update_cache=True`` makes the fused decode kernel write the new K/V into the
    paged cache itself, so the caller does no scatter. For block KV the kernel wants
    physical slots, and the shape depends on the head count
    (``attention_block_tkg`` docstring):

    * ``kv_heads == 1``: ``[B, S_tkg]``
    * ``kv_heads >  1``: ``[B, kv_heads, S_tkg]``

    A slot must index the SAME flat pool that ``build_per_head_block_table`` targets:
    the kernel folds the 4D cache to ``num_blocks * kv_heads`` blocks, so the global
    block of ``(base_block, h)`` is ``base_block * kv_heads + h``. A slot is then
    ``global_block * block_size + position_in_block``. This function and
    ``build_per_head_block_table`` (in ``qwen3_vl.utils.decode_kv``) must agree.

    Args:
        slot_mapping: ``[B * S_tkg]`` flat slots, ``base_block * block_size + pos``.
        block_size: Tokens in one cache block.
        kv_heads: KV heads of this rank.
        batch_size: Requests of this decode step.

    Returns:
        ``[B, kv_heads, S_tkg]`` slots, or ``[B, S_tkg]`` when ``kv_heads == 1``.

    Example:
        >>> import torch
        >>> # 1 request, 1 token at base block 2 position 3, block_size 4
        >>> slots = torch.tensor([2 * 4 + 3])
        >>> build_per_head_slot_mapping(slots, block_size=4, kv_heads=2, batch_size=1)
        tensor([[[19],
                 [23]]])
    """
    if kv_heads == 1:
        return slot_mapping.view(batch_size, -1)

    base_block = torch.div(slot_mapping, block_size, rounding_mode="floor")
    position = slot_mapping - base_block * block_size

    kv_h = torch.arange(
        kv_heads, dtype=slot_mapping.dtype, device=slot_mapping.device
    ).view(1, kv_heads)
    # [B*S, kv_heads] -> global slot of each (token, head)
    global_block = base_block.unsqueeze(1) * kv_heads + kv_h
    global_slot = global_block * block_size + position.unsqueeze(1)
    # [B*S, kv_heads] -> [B, kv_heads, S_tkg]
    return global_slot.view(batch_size, -1, kv_heads).transpose(1, 2).contiguous()
