# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 hybrid text model for Neuron.

The model mixes two token mixers. A Gated DeltaNet layer runs linear attention
and keeps a fixed-size recurrent state. A full-attention layer runs grouped-query
attention with an output gate and keeps a key/value cache. ``config.layer_types``
selects the mixer of each layer. The checkpoint pattern gives one full-attention
layer in each group of four layers.

The projections, the Gated DeltaNet convolution and delta rule, and the MLP use
NKI kernels when the kernel can run the call. Other calls use torch.

The projections that read the same input join into one weight at load time:
q (with the gate), k and v in full attention, and in_proj_qkv and in_proj_z in
the Gated DeltaNet. In prefill, one QKV kernel call runs each joined weight,
together with the input RMSNorm of the layer. The MLP kernel applies the
post-attention RMSNorm. The decoder layer passes both norms to the kernels, so a
norm runs outside a kernel only on the torch path.

The Gated DeltaNet core takes its state as an explicit input. It can return the
final state or token checkpoints. A later cache integration binds these values
to runner-owned pages.

Each linear parameter uses ``[in_features, out_features]`` because its
projection computes ``x @ weight``. The checkpoint stores each tensor as
``[out_features, in_features]``. Each loader sets
``is_storage_transposed=True`` to transpose the checkpoint tensor during load.
"""

import logging

import torch
import torch.nn.functional as F
from nkilib.core.qkv.qkv import SEQLEN_THRESHOLD_FOR_QKV_CTE
from nkilib.core.utils.common_types import (
    ComputationMode,
    NormType,
    QKVOutputLayout,
    QuantizationType,
)
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.kv_cache_interface import MambaAttentionBackendEnum

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.functional.attention.o_proj import (
    _can_use_kernel as _o_proj_kernel_can_run,
)
from vllm_neuron.functional.attention.qkv import (
    _can_use_qkv_kernel as _qkv_kernel_can_run,
)
from vllm_neuron.functional.gated_delta_net import (
    build_gdn_mask_pack,
    chunked_gated_delta_rule,
    gdn_conv1d,
    gdn_conv1d_reject_reason,
    gdn_decode,
    gdn_decode_reject_reason,
    gdn_prefill,
    gdn_prefill_reject_reason,
    recurrent_gated_delta_rule,
)
from vllm_neuron.functional.mlp import _can_fuse_decode_norm
from vllm_neuron.functional.mlp import _can_use_kernel as can_use_mlp_kernel
from vllm_neuron.functional.recurrent_state import (
    prepare_mamba_state_cache_access,
    read_mamba_state_pages_grouped,
    write_mamba_state_pages_grouped,
)
from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    MambaStateViewLayout,
    RecurrentLayerSpec,
    paged_slot_indices,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5TextConfig

logger = logging.getLogger(__name__)

# The checkpoint stores the text weights under this prefix.
HF_TEXT_PREFIX = "model.language_model"

# Tensor groups that this port does not load. The vision tower and the
# multi-token-prediction head are out of scope for the baseline.
UNUSED_HF_PREFIXES = ("model.visual.", "mtp.")

# TODO: Temporary. The device profile does not attribute ops to layers, so this
# log shows the NKI or torch path of each operation. See the TODO in
# vllm_neuron/functional/gated_delta_net.py.
_KERNEL_PATHS_LOGGED: set[tuple[str, str, str]] = set()


def build_recurrent_token_mask(
    position_ids: torch.Tensor,
    state_write_valid: torch.Tensor,
) -> torch.Tensor:
    """Mark real tokens in a runner-padded Qwen3.8 batch."""
    if position_ids.ndim != 2:
        raise ValueError("Qwen3.8 position IDs must have shape [batch, sequence].")
    if state_write_valid.ndim != 2 or state_write_valid.shape[1] == 0:
        raise ValueError(
            "The recurrent write-valid mask must have shape [batch, states]."
        )
    if state_write_valid.shape[0] != position_ids.shape[0]:
        raise ValueError(
            "The recurrent write-valid mask and position IDs must have the same batch."
        )

    active_rows = state_write_valid.to(torch.bool).any(dim=1, keepdim=True)
    if position_ids.shape[1] == 1:
        return active_rows

    # Real positions are consecutive. The runner can repeat the final position
    # or use zero for padding. Compare each column with the expected sequence.
    # For example, [7, 8, 9, 9] gives [True, True, True, False].
    token_index = torch.arange(
        position_ids.shape[1],
        dtype=position_ids.dtype,
        device=position_ids.device,
    )
    expected_positions = position_ids[:, :1] + token_index[None, :]
    return (position_ids == expected_positions) & active_rows


def build_qwen3_5_kv_spec(config: Qwen3_5TextConfig, tp_size: int) -> KVSpec:
    """Build the per-rank full-attention and recurrent cache specification."""
    if tp_size <= 0:
        raise ValueError(f"The tensor-parallel size is {tp_size}. Use a positive size.")
    for name, count in (
        ("linear_num_key_heads", config.linear_num_key_heads),
        ("linear_num_value_heads", config.linear_num_value_heads),
    ):
        if count % tp_size:
            raise ValueError(
                f"{name}={count} does not divide by tensor-parallel size "
                f"{tp_size}. Select a compatible size."
            )
    if tp_size < config.num_key_value_heads and config.num_key_value_heads % tp_size:
        raise ValueError(
            f"num_key_value_heads={config.num_key_value_heads} does not divide by "
            f"tensor-parallel size {tp_size}. Select a compatible size."
        )

    local_key_heads = config.linear_num_key_heads // tp_size
    local_value_heads = config.linear_num_value_heads // tp_size
    local_conv_width = (
        2 * local_key_heads * config.linear_key_head_dim
        + local_value_heads * config.linear_value_head_dim
    )
    local_kv_heads = (
        1
        if tp_size >= config.num_key_value_heads
        else config.num_key_value_heads // tp_size
    )

    layers = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == FULL_ATTENTION:
            layers.append(
                LayerSpec(
                    name=f"layers.{layer_idx}.self_attn",
                    num_kv_heads=local_kv_heads,
                    head_size=config.head_dim,
                    dtype=config.torch_dtype,
                )
            )
            continue

        layers.append(
            RecurrentLayerSpec(
                name=f"layers.{layer_idx}.linear_attn",
                state_shapes=(
                    (config.linear_conv_kernel_dim - 1, local_conv_width),
                    (
                        local_value_heads,
                        config.linear_value_head_dim,
                        config.linear_key_head_dim,
                    ),
                ),
                state_dtypes=(config.torch_dtype, config.mamba_ssm_dtype),
                state_names=("conv", "recurrent"),
                state_backend_type=MambaAttentionBackendEnum.GDN_ATTN,
                state_num_prefill_checkpoint_blocks=1,
                state_prefill_checkpoint_alignment=64,
            )
        )
    return KVSpec(layers=layers)


# Dynamo cannot trace a logger call.
@torch._dynamo.assume_constant_result
def _log_kernel_path(op: str, path: str, detail: str) -> None:
    """Log the path of one operation, NKI or torch, one time for each decision."""
    key = (op, path, detail)
    if key in _KERNEL_PATHS_LOGGED:
        return
    _KERNEL_PATHS_LOGGED.add(key)
    logger.info("[qwen3_5] %s: %s (%s)", op, path, detail)


# ── Weight loaders ────────────────────────────────────────────────────────────


def segmented_sharding_weight_loader(
    segment_sizes: list[int],
    num_shards: int,
    storage_dim: int = 0,
    is_storage_transposed: bool = False,
) -> SafetensorsWeightLoader:
    """Shard a checkpoint tensor that concatenates segments of different widths.

    A plain shard of in_proj_qkv would cut across the q/k/v boundaries, so shard
    each segment along ``storage_dim`` and join the parts of the rank in order.
    ``is_storage_transposed`` returns the ``[in, out]`` parameter layout.
    """

    def transform(slices: list["object"], rank: int) -> torch.Tensor:
        assert len(slices) == 1, (
            "segmented_sharding_weight_loader() supports a single tensor only"
        )
        slice_obj = slices[0]
        num_dims = len(slice_obj.get_shape())

        parts = []
        offset = 0
        for size in segment_sizes:
            per_shard = size // num_shards
            start = offset + (rank % num_shards) * per_shard
            index = [slice(None)] * num_dims
            index[storage_dim] = slice(start, start + per_shard)
            parts.append(slice_obj[tuple(index)])
            offset += size

        result = torch.cat(parts, dim=storage_dim)
        return result.T if is_storage_transposed else result

    return SafetensorsWeightLoader(transform=transform)


def in_proj_qkvz_weight_loader(
    key_dim: int, value_dim: int, num_shards: int
) -> SafetensorsWeightLoader:
    """Join the in_proj_qkv and in_proj_z shards of a rank into one weight.

    in_proj_qkv takes the segmented shard, and in_proj_z a column shard. The result
    is ``[hidden, (2 * key_dim + 2 * value_dim) / num_shards]``.
    """
    qkv_loader = segmented_sharding_weight_loader(
        segment_sizes=[key_dim, key_dim, value_dim],
        num_shards=num_shards,
        storage_dim=0,
        is_storage_transposed=True,
    )
    z_loader = sharding_weight_loader(
        shard_dim=1,
        shard_size=value_dim // num_shards,
        num_shards=num_shards,
        is_storage_transposed=True,
    )

    def transform(slices: list["object"], rank: int) -> torch.Tensor:
        if len(slices) != 2:
            raise ValueError(
                f"The in_proj_qkvz loader got {len(slices)} slices. Map the weight "
                "to the in_proj_qkv and the in_proj_z tensors."
            )
        qkv, z = slices
        return torch.cat(
            [qkv_loader.load([qkv], rank), z_loader.load([z], rank)], dim=1
        )

    return SafetensorsWeightLoader(transform=transform)


# ── Normalisation ─────────────────────────────────────────────────────────────


class Qwen3_5RMSNorm(nn.Module):
    """Zero-centered RMS norm.

    The scale is ``1 + weight`` and the checkpoint stores ``weight`` around zero.
    A standard RMS norm multiplies by ``weight`` alone, so it gives a wrong
    result on this checkpoint.
    """

    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)


class Qwen3_5RMSNormGated(nn.Module):
    """RMS norm with a gate. A Gated DeltaNet layer applies it to the mixer output.

    This norm scales by ``weight`` alone. The checkpoint stores ``weight`` around
    one, so it is not zero-centered.
    """

    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.weight * out.to(in_dtype)
        out = out * F.silu(gate.float())
        return out.to(in_dtype)


# ── Rotary embedding ──────────────────────────────────────────────────────────


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the rotary embedding to the first ``cos.shape[-1]`` channels.

    The model rotates a part of each head. The remaining channels pass through.
    ``q`` and ``k`` use the layout ``[batch, seq, heads, head_dim]``.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_out = q_rot * cos + rotate_half(q_rot) * sin
    k_out = k_rot * cos + rotate_half(k_rot) * sin
    return (
        torch.cat([q_out, q_pass], dim=-1),
        torch.cat([k_out, k_pass], dim=-1),
    )


class Qwen3_5RotaryEmbedding(nn.Module):
    """mRoPE for the text path.

    The model carries three position axes for time, height and width. A text-only
    request repeats one axis three times, so the recomposition returns the time
    axis. The class keeps the three axes, because an image request needs them.
    """

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.mrope_section = list(config.mrope_section)
        dim = config.rotary_dim
        # Model Runner constructs the module under a meta-device context. These
        # derived constants do not come from the checkpoint, so keep them real
        # on CPU until the runner moves the loaded model to the Neuron device.
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build one selection mask for each position axis. The reference writes
        # the axes into one block with a strided in-place assignment. An in-place
        # write to a strided view of a device tensor does not lower, so select
        # with a mask product instead. The result is the same block.
        num_freqs = inv_freq.shape[0]
        axis_of = [0] * num_freqs
        for axis, offset in enumerate((1, 2), start=1):
            for index in range(offset, self.mrope_section[axis] * 3, 3):
                if index < num_freqs:
                    axis_of[index] = axis
        masks = torch.zeros(3, num_freqs, dtype=torch.float32, device="cpu")
        for index, axis in enumerate(axis_of):
            masks[axis, index] = 1.0
        self.register_buffer("axis_masks", masks, persistent=False)

    def _recompose(self, freq: torch.Tensor) -> torch.Tensor:
        """Select the three position axes into one frequency block.

        ``freq`` uses shape ``[3, batch, seq, num_freqs]``. Each output channel
        reads one axis, so a sum of three masked axes gives the exact block.
        """
        masks = self.axis_masks.to(freq.dtype)
        out = freq[0] * masks[0] + freq[1] * masks[1] + freq[2] * masks[2]
        return torch.cat((out, out), dim=-1)

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``cos`` and ``sin`` of shape ``[batch, seq, rotary_dim]``.

        ``position_ids`` uses shape ``[batch, seq]`` for a text request, or
        ``[3, batch, seq]`` for a request that holds image or video tokens.
        """
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        inv_freq = self.inv_freq[None, None, :, None].expand(
            3, position_ids.shape[1], -1, 1
        )
        freqs = (inv_freq.float() @ position_ids[:, :, None, :].float()).transpose(2, 3)
        cos = self._recompose(freqs.cos())
        sin = self._recompose(freqs.sin())
        return cos.to(dtype), sin.to(dtype)


# ── Projections ───────────────────────────────────────────────────────────────

# The kernels take each head dimension and each scale on the 128 partitions.
_PMAX = 128


def _kernel_norm(norm: nn.Module | None) -> Qwen3_5RMSNorm | None:
    """Return ``norm`` when a kernel can apply it, or None."""
    return norm if isinstance(norm, Qwen3_5RMSNorm) else None


def _kernel_gamma(norm: Qwen3_5RMSNorm, dtype: torch.dtype) -> torch.Tensor:
    """Return the norm scale for a kernel. The norm is zero-centered."""
    return (1.0 + norm.weight.float()).to(dtype).view(1, -1)


def _qkv_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    op: str,
    norm: nn.Module | None = None,
    normed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``norm(x) @ weight``. The prefill QKV kernel also applies ``norm``.

    The torch path uses ``normed`` when the caller already holds ``norm(x)``.
    """
    fused_norm = _kernel_norm(norm)
    if norm is not None and fused_norm is None:
        # The kernel applies only the RMS norm of this model. Apply another norm
        # first.
        x = norm(x) if normed is None else normed
        norm = normed = None
    batch, seq_len = x.shape[0], x.shape[1]
    shape = f"T={batch * seq_len} H={x.shape[-1]} N={weight.shape[1]}"
    if not can_run_kernel(x):
        reason = "NKI kernels are disabled or the tensor is on CPU"
    elif seq_len <= SEQLEN_THRESHOLD_FOR_QKV_CTE and batch * seq_len <= _PMAX:
        # NF.qkv_proj then runs qkv_tkg, which adds its partial sums in BF16 and
        # doubles the rounding error of a BF16 matmul.
        reason = "qkv_tkg loses BF16 precision at this shape"
    elif not _qkv_kernel_can_run(
        x,
        weight,
        None,
        None,
        None,
        QKVOutputLayout.BSD,
        NormType.NO_NORM if fused_norm is None else NormType.RMS_NORM,
        None,
    ):
        reason = "the shape is outside the kernel limits"
    else:
        reason = None
    if reason is None:
        norm_args = {}
        if fused_norm is not None:
            norm_args = dict(
                norm_weights=_kernel_gamma(fused_norm, x.dtype),
                norm_type=NormType.RMS_NORM,
                eps=fused_norm.eps,
            )
        _log_kernel_path(op, "NKI qkv" + (" + norm" if fused_norm else ""), shape)
        return NF.qkv_proj(x, weight, **norm_args).to(x.dtype)
    _log_kernel_path(op, "torch", f"{shape}; {reason}")
    if normed is None:
        normed = x if norm is None else norm(x)
    return normed @ weight


def _o_proj_heads(head_dim: int) -> int:
    """Return how many kernel heads hold one head of ``head_dim``.

    The o_proj kernel takes a head dimension of at most 128, so a 256-wide head
    goes in as two 128-wide heads.
    """
    return head_dim // _PMAX if head_dim > _PMAX and head_dim % _PMAX == 0 else 1


def _o_proj_input(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """Reshape ``[B, S, heads * head_dim]`` to the kernel layout ``[1, N, D, T]``."""
    batch, seq_len, _ = x.shape
    return (
        x.reshape(batch * seq_len, heads, head_dim)
        .permute(1, 2, 0)
        .unsqueeze(0)
        .contiguous()
    )


def _output_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    op: str,
    heads: int,
    head_dim: int,
    one_token_fp32: bool = False,
) -> torch.Tensor:
    """Return ``x @ weight`` on the o_proj kernel, or in torch.

    ``x`` is ``[B, S, heads * head_dim]``, and the weight rows follow the same
    head order.
    """
    batch, seq_len, _ = x.shape
    split = _o_proj_heads(head_dim)
    heads, head_dim = heads * split, head_dim // split
    shape = f"T={batch * seq_len} N={heads} D={head_dim} H={weight.shape[1]}"
    if batch * seq_len == 1 and one_token_fp32:
        # The caller enables this path for one token with four or more TP ranks.
        # BF16 rounding of each rank-local partial failed the TP4 single-token
        # accuracy tolerance. Keep the partial sums in FP32 through all-reduce.
        _log_kernel_path(op, "torch FP32", f"{shape}; one-token precision path")
        return x.float() @ weight.float()
    reason = None
    if not can_run_kernel(x):
        reason = "NKI kernels are disabled or the tensor is on CPU"
    else:
        active = _o_proj_input(x, heads, head_dim)
        if not _o_proj_kernel_can_run(active, weight):
            reason = "the shape is outside the kernel limits"
    if reason is not None:
        _log_kernel_path(op, "torch", f"{shape}; {reason}")
        return x @ weight
    _log_kernel_path(op, "NKI o_proj", shape)
    out = NF.o_proj(active, weight, None)
    return out.reshape(batch, seq_len, -1).to(x.dtype)


# ── Full-attention layer ──────────────────────────────────────────────────────


def build_causal_mask(
    query_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device_anchor: torch.Tensor,
) -> torch.Tensor:
    """Return an additive causal mask of shape ``[1, 1, query_len, kv_len]``.

    The query at index ``i`` attends to the key at index ``j`` when
    ``j <= kv_len - query_len + i``.
    """
    device = device_anchor.device
    past = kv_len - query_len
    rows = torch.arange(query_len, device=device).unsqueeze(1) + past
    cols = torch.arange(kv_len, device=device).unsqueeze(0)
    blocked = cols > rows
    # Keep a rank-local tensor input in this graph. Explicit recurrent compile
    # regions can otherwise isolate this constant-shape mask into a zero-input
    # native graph, whose output allocation defaults to Neuron device 0.
    mask = device_anchor.new_zeros((query_len, kv_len), dtype=dtype)
    mask = mask.masked_fill(blocked, torch.finfo(dtype).min)
    return mask[None, None, :, :]


class Qwen3_5Attention(nn.Module):
    """Grouped-query attention with per-head QK norm and an output gate.

    q_proj emits ``2 * head_dim`` channels per head: the query, then the gate. q, k
    and v join into one ``qkv_proj`` weight, so in prefill one QKV kernel call
    runs all three with the input RMSNorm.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_output_gate = config.attn_output_gate

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_heads_per_rank = self.num_heads // self.world_size

        # Replicate the key/value heads when the group holds more ranks than heads.
        if self.world_size >= self.num_kv_heads:
            self.num_kv_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_kv_heads
        else:
            self.num_kv_heads_per_rank = self.num_kv_heads // self.world_size
            self.num_kv_replicas = 1
        self.num_kv_groups = self.num_heads_per_rank // self.num_kv_heads_per_rank

        gate_factor = 2 if self.attn_output_gate else 1
        q_out = self.num_heads_per_rank * self.head_dim * gate_factor
        kv_out = self.num_kv_heads_per_rank * self.head_dim
        o_in = self.num_heads_per_rank * self.head_dim
        self.qkv_split = [q_out, kv_out, kv_out]

        self._create_projection_weights(q_out, kv_out, o_in)

        # The QK norm runs on the head dimension, so every rank holds it whole.
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

        self._setup_weight_loaders(q_out, kv_out, o_in)

    def _create_projection_weights(
        self,
        q_out: int,
        kv_out: int,
        o_in: int,
    ) -> None:
        """Create the BF16 attention projection weights."""
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_out + 2 * kv_out, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_in, self.hidden_size, dtype=self.dtype)
        )

    def _project_qkv(
        self, x: torch.Tensor, norm: nn.Module | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the q (with the gate), k and v projections of ``norm(x)``."""
        out = _qkv_projection(x, self.qkv_proj_weight, "Attention qkv", norm)
        return out.split(self.qkv_split, dim=-1)

    def _project_output(self, x: torch.Tensor) -> torch.Tensor:
        """Return the o_proj projection of ``x``."""
        return _output_projection(
            x,
            self.o_proj_weight,
            "Attention o_proj",
            self.num_heads_per_rank,
            self.head_dim,
            one_token_fp32=self.world_size >= 4,
        )

    def _setup_weight_loaders(self, q_out: int, kv_out: int, o_in: int) -> None:
        # The query projection groups the query and the gate for each head, so a
        # shard of whole heads stays contiguous. Several ranks read the same
        # key/value head when replication applies.
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=q_out,
                kv_size=kv_out,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )

        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=o_in,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def _write_paged_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Scatter this step's K/V into scheduler-selected physical slots."""
        if self.k_cache is None or self.v_cache is None:
            raise ValueError(
                "The full-attention cache is not bound. Bind the K/V cache first."
            )
        if self.k_cache.ndim != 4 or self.v_cache.ndim != 4:
            raise ValueError(
                "The full-attention cache has the wrong rank. "
                "Use [blocks, heads, block_size, head_dim]."
            )

        block_indices, position_indices = paged_slot_indices(
            slot_mapping,
            block_size,
            self.k_cache.shape[0],
        )
        num_tokens = slot_mapping.shape[0]
        head_indices = torch.arange(
            self.num_kv_heads_per_rank,
            dtype=torch.long,
            device=key.device,
        ).repeat(num_tokens)
        block_indices = block_indices.repeat_interleave(self.num_kv_heads_per_rank)
        position_indices = position_indices.repeat_interleave(
            self.num_kv_heads_per_rank
        )

        self.k_cache.index_put_(
            (block_indices, head_indices, position_indices),
            key.reshape(-1, self.head_dim).to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_indices, head_indices, position_indices),
            value.reshape(-1, self.head_dim).to(self.v_cache.dtype),
        )

    def _read_paged_cache(
        self,
        block_table: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather each request's logical prefix and its causal validity mask."""
        if self.k_cache is None or self.v_cache is None:
            raise ValueError(
                "The full-attention cache is not bound. Bind the K/V cache first."
            )

        valid_blocks = (block_table > 0) & (block_table < self.k_cache.shape[0])
        safe_blocks = torch.where(
            valid_blocks,
            block_table,
            torch.zeros_like(block_table),
        ).to(torch.long)
        batch, max_blocks = safe_blocks.shape
        max_context = max_blocks * block_size

        key = self.k_cache[safe_blocks]
        value = self.v_cache[safe_blocks]
        key = key.permute(0, 1, 3, 2, 4).reshape(
            batch,
            max_context,
            self.num_kv_heads_per_rank,
            self.head_dim,
        )
        value = value.permute(0, 1, 3, 2, 4).reshape(
            batch,
            max_context,
            self.num_kv_heads_per_rank,
            self.head_dim,
        )

        valid_tokens = valid_blocks.repeat_interleave(block_size, dim=1)
        key_positions = torch.arange(
            max_context,
            dtype=positions.dtype,
            device=positions.device,
        )
        allowed = valid_tokens[:, None, None, :] & (
            key_positions[None, None, None, :] <= positions[:, None, :, None]
        )
        return key, value, allowed

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        attn_metadata: dict[str, object] | None = None,
        positions: torch.Tensor | None = None,
        input_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        query, key, value = self._project_qkv(hidden_states, input_norm)
        if self.attn_output_gate:
            query = query.view(batch, seq_len, self.num_heads_per_rank, -1)
            query, gate = query.chunk(2, dim=-1)
            gate = gate.reshape(batch, seq_len, -1)
        else:
            query = query.view(batch, seq_len, self.num_heads_per_rank, self.head_dim)
            gate = None

        query = self.q_norm(query)
        key = self.k_norm(
            key.view(batch, seq_len, self.num_kv_heads_per_rank, self.head_dim)
        )
        value = value.view(batch, seq_len, self.num_kv_heads_per_rank, self.head_dim)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)

        paged_allowed = None
        segmented_output = None
        if attn_metadata is not None:
            if positions is None:
                raise ValueError(
                    "Paged full attention has no query positions. "
                    "Pass one position for each query token."
                )
            slot_mapping = attn_metadata["slot_mapping"]
            block_size = attn_metadata["block_size"]
            block_table = attn_metadata["block_table_tensor"]
            cached_seq_len = attn_metadata.get("cached_seq_len")
            kv_segment_size = attn_metadata.get("kv_segment_size")
            if not isinstance(slot_mapping, torch.Tensor):
                raise TypeError(
                    "The full-attention slot mapping is not a tensor. Pass a tensor."
                )
            if not isinstance(block_size, int):
                raise TypeError(
                    "The full-attention block size is not an integer. Pass an integer."
                )
            if not isinstance(block_table, torch.Tensor):
                raise TypeError(
                    "The full-attention block table is not a tensor. Pass a tensor."
                )
            if kv_segment_size is not None and not isinstance(kv_segment_size, int):
                raise TypeError(
                    "The segmented attention size is not an integer. "
                    "Pass kv_segment_size as an integer."
                )
            if kv_segment_size:
                if not isinstance(cached_seq_len, torch.Tensor):
                    raise TypeError(
                        "The segmented attention cache length is not a tensor. "
                        "Pass cached_seq_len as a tensor."
                    )
            self._write_paged_cache(key, value, slot_mapping, block_size)
            if kv_segment_size:
                if batch != 1:
                    raise ValueError(
                        "Segmented prefill requires one active request. "
                        "Schedule one prefill request per step."
                    )
                # Move the query heads to the kernel's leading axis.
                segmented_query = (
                    query.permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(
                        self.num_heads_per_rank,
                        seq_len,
                        self.head_dim,
                    )
                )
                segmented_output = NF.segmented_attention(
                    segmented_query,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    block_tables=block_table,
                    prior_tokens=cached_seq_len,
                    block_size=block_size,
                    kv_segment_size=kv_segment_size,
                    scale=self.scaling,
                    tp_q=True,
                    tp_out=True,
                )
            else:
                key, value, paged_allowed = self._read_paged_cache(
                    block_table,
                    positions,
                    block_size,
                )

        if segmented_output is not None:
            # Restore the model layout before the output gate and projection.
            attn_out = (
                segmented_output.reshape(
                    batch,
                    self.num_heads_per_rank,
                    self.head_dim,
                    seq_len,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
                .reshape(batch, seq_len, -1)
            )
        else:
            # Move the head axis next to the batch axis for the attention matmuls.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.num_kv_groups > 1:
                key = key.repeat_interleave(self.num_kv_groups, dim=1)
                value = value.repeat_interleave(self.num_kv_groups, dim=1)

            scores = query @ key.transpose(2, 3) * self.scaling
            if paged_allowed is None:
                # Clamp before the cast. A float32 mask cast to bfloat16 gives
                # -inf, which can produce NaN for a fully masked padding row.
                mask = attention_mask.clamp(min=torch.finfo(scores.dtype).min)
                scores = scores + mask.to(scores.dtype)
            else:
                scores = scores.masked_fill(
                    ~paged_allowed,
                    torch.finfo(scores.dtype).min,
                )
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_out = weights @ value
            attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, -1)

        if gate is not None:
            attn_out = attn_out * torch.sigmoid(gate)

        out = self._project_output(attn_out)
        # The output projection is row-parallel, so each rank holds a partial sum.
        # Assign the result of ``all_reduce``. Do not call it as a statement:
        # ``GroupCoordinator.all_reduce`` is out of place on the custom-operator
        # path, so a statement drops the reduction without an error. The
        # communicator path works in place and returns the same tensor, so the
        # assignment is correct on both paths.
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out.to(hidden_states.dtype)


# ── Gated DeltaNet layer ──────────────────────────────────────────────────────


class Qwen3_5GatedDeltaNet(nn.Module):
    """Linear-attention mixer with a short convolution and a gated delta rule.

    The state has a fixed size: the last ``conv_kernel_size - 1`` tokens for the
    convolution, and one matrix per value head for the delta rule. in_proj_qkv and
    in_proj_z join into one ``in_proj_qkvz`` weight for one prefill QKV kernel call
    with the input RMSNorm. in_proj_a and in_proj_b (one column per value head)
    run in torch.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.state_dtype = config.mamba_ssm_dtype
        self.hidden_size = config.hidden_size
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads_per_rank = self.num_k_heads // self.world_size
        self.num_v_heads_per_rank = self.num_v_heads // self.world_size
        # Each key head serves this many value heads.
        self.num_v_per_k = self.num_v_heads // self.num_k_heads

        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.key_dim_per_rank = self.num_k_heads_per_rank * self.head_k_dim
        self.value_dim_per_rank = self.num_v_heads_per_rank * self.head_v_dim
        self.conv_dim_per_rank = 2 * self.key_dim_per_rank + self.value_dim_per_rank

        self._create_projection_weights()
        # The norm runs on one value head, so every rank holds it whole.
        self.norm = Qwen3_5RMSNormGated(
            self.head_v_dim, config.rms_norm_eps, self.dtype
        )
        # Constant masks for gdn_cte.
        self.register_buffer("gdn_mask_pack", build_gdn_mask_pack(), persistent=False)

        self.state_cache_bases: tuple[torch.Tensor, ...] = ()
        self.state_cache_layout: MambaStateViewLayout = ()
        self.state_cache_group: int | None = None

        self._setup_weight_loaders()

    def _create_projection_weights(self) -> None:
        """Create the BF16 GDN projection weights."""
        self.in_proj_qkvz_weight = nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.conv_dim_per_rank + self.value_dim_per_rank,
                dtype=self.dtype,
            )
        )
        self.in_proj_b_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.num_v_heads_per_rank, dtype=self.dtype)
        )
        self.in_proj_a_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.num_v_heads_per_rank, dtype=self.dtype)
        )
        # Keep the checkpoint convolution layout.
        self.conv1d_weight = nn.Parameter(
            torch.empty(
                self.conv_dim_per_rank,
                1,
                self.conv_kernel_size,
                dtype=self.dtype,
            )
        )
        self.out_proj_weight = nn.Parameter(
            torch.empty(self.value_dim_per_rank, self.hidden_size, dtype=self.dtype)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_v_heads_per_rank, dtype=torch.float32)
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads_per_rank, dtype=torch.float32)
        )

    def _project_inputs(
        self, x: torch.Tensor, norm: nn.Module | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the in_proj_qkv, in_proj_z, in_proj_b and in_proj_a outputs.

        Each projection reads ``norm(x)``.
        """
        # in_proj_a and in_proj_b run in torch, so they need the normed input.
        normed = x if norm is None else norm(x)
        qkvz = _qkv_projection(x, self.in_proj_qkvz_weight, "GDN in_proj", norm, normed)
        mixed_qkv, z = qkvz.split(
            [self.conv_dim_per_rank, self.value_dim_per_rank], dim=-1
        )
        return (
            mixed_qkv,
            z,
            normed @ self.in_proj_b_weight,
            normed @ self.in_proj_a_weight,
        )

    def _project_output(self, x: torch.Tensor) -> torch.Tensor:
        """Return the out_proj projection of ``x``."""
        # The weight rows are ordered value head by value head.
        return _output_projection(
            x,
            self.out_proj_weight,
            "GDN out_proj",
            self.num_v_heads_per_rank,
            self.head_v_dim,
        )

    def _setup_weight_loaders(self) -> None:
        segments = [self.key_dim, self.key_dim, self.value_dim]
        set_weight_loader(
            self.in_proj_qkvz_weight,
            in_proj_qkvz_weight_loader(self.key_dim, self.value_dim, self.world_size),
        )
        # The convolution shares the q/k/v segment order of the input projection.
        set_weight_loader(
            self.conv1d_weight,
            segmented_sharding_weight_loader(
                segment_sizes=segments,
                num_shards=self.world_size,
                storage_dim=0,
                is_storage_transposed=False,
            ),
        )
        for param in (self.in_proj_b_weight, self.in_proj_a_weight):
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=self.num_v_heads_per_rank,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                ),
            )
        set_weight_loader(
            self.out_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.value_dim_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        for param in (self.dt_bias, self.A_log):
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.num_v_heads_per_rank,
                    num_shards=self.world_size,
                ),
            )

    def _apply_conv_with_state(
        self,
        mixed_qkv: torch.Tensor,
        state: torch.Tensor | None,
        token_valid_mask: torch.Tensor | None = None,
        return_state_checkpoints: bool = False,
        checkpoint_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the depthwise convolution and return the next state.

        ``mixed_qkv`` uses the layout ``[batch, conv_dim, seq]``.
        """
        _, _, seq_len = mixed_qkv.shape
        window = self.conv_kernel_size - 1

        if state is not None:
            # Prepend the saved window, then convolve with no padding.
            padded = torch.cat([state.to(mixed_qkv.dtype), mixed_qkv], dim=-1)
        else:
            padded = F.pad(mixed_qkv, (window, 0))

        op = "GDN conv1d decode" if seq_len == 1 else "GDN conv1d prefill"
        reject_reason = gdn_conv1d_reject_reason(
            mixed_qkv, state, self.conv_kernel_size
        )
        shape = f"B={mixed_qkv.shape[0]} C={mixed_qkv.shape[1]} T={seq_len}"
        if reject_reason is None:
            kernel = "gdn_conv1d_decode" if seq_len == 1 else "gdn_conv1d_prefill"
            _log_kernel_path(op, f"NKI {kernel}", shape)
            out = gdn_conv1d(mixed_qkv, self.conv1d_weight.squeeze(1), state)
        else:
            _log_kernel_path(op, "torch", f"{shape}; {reject_reason}")
            # Torch path.
            weight = self.conv1d_weight.squeeze(1).unsqueeze(0)  # [1, conv_dim, k]
            out = padded[:, :, 0:seq_len] * weight[:, :, 0:1]
            for k in range(1, self.conv_kernel_size):
                out = out + padded[:, :, k : k + seq_len] * weight[:, :, k : k + 1]
            out = F.silu(out)

        if return_state_checkpoints and checkpoint_offsets is not None:
            raise ValueError(
                "Choose speculative checkpoints or an internal prefill "
                "checkpoint, not both."
            )

        if return_state_checkpoints:
            if token_valid_mask is None:
                next_state = torch.stack(
                    [
                        padded[:, :, position + 1 : position + 1 + window]
                        for position in range(seq_len)
                    ],
                    dim=1,
                )
            else:
                # A padded position repeats the last real checkpoint.
                completed_tokens = token_valid_mask.to(torch.long).cumsum(dim=1)
                state_indices = completed_tokens[:, :, None] + torch.arange(
                    window,
                    dtype=torch.long,
                    device=padded.device,
                )
                state_indices = state_indices[:, :, None, :].expand(
                    -1,
                    -1,
                    padded.shape[1],
                    -1,
                )
                expanded = padded[:, None, :, :].expand(
                    -1,
                    seq_len,
                    -1,
                    -1,
                )
                next_state = torch.gather(expanded, 3, state_indices)
        else:
            if token_valid_mask is None:
                final_state = padded[:, :, -window:]
            else:
                # Padding is always a suffix of each request row. The
                # convolution output for the real prefix is already correct,
                # but the physical end of ``padded`` contains fake token
                # projections. Gather the window ending at each row's last
                # real token instead.
                valid_lengths = token_valid_mask.sum(dim=1, dtype=torch.long)
                state_indices = valid_lengths[:, None] + torch.arange(
                    window,
                    dtype=torch.long,
                    device=padded.device,
                )
                state_indices = state_indices[:, None, :].expand(
                    -1,
                    padded.shape[1],
                    -1,
                )
                final_state = torch.gather(padded, 2, state_indices)

            if checkpoint_offsets is None:
                next_state = final_state
            else:
                offsets = checkpoint_offsets.reshape(-1).to(
                    device=padded.device,
                    dtype=torch.long,
                )
                if offsets.shape != padded.shape[:1]:
                    raise ValueError(
                        "GDN checkpoint offsets must contain one value per request."
                    )
                checkpoint_indices = offsets[:, None] + torch.arange(
                    window,
                    dtype=torch.long,
                    device=padded.device,
                )
                checkpoint_indices = checkpoint_indices[:, None, :].expand(
                    -1,
                    padded.shape[1],
                    -1,
                )
                checkpoint_state = torch.gather(
                    padded,
                    2,
                    checkpoint_indices,
                )
                next_state = torch.stack((checkpoint_state, final_state), dim=1)

        return out, next_state

    def paged_state_read_request(
        self,
        attn_metadata: dict[str, object],
    ) -> tuple[MambaStateViewLayout, torch.Tensor, torch.Tensor | None]:
        """Return the cache layout, selected read pages, and cold-state mask."""
        if not self.state_cache_bases:
            raise ValueError(
                "The GDN recurrent cache is not bound. Bind the state cache first."
            )
        page_ids = attn_metadata.get("recurrent_state_read_page_ids")
        cold = attn_metadata.get("recurrent_state_cold")
        if not isinstance(page_ids, torch.Tensor):
            raise TypeError(
                "The GDN read page IDs are not a tensor. Pass recurrent read pages."
            )
        if cold is not None and not isinstance(cold, torch.Tensor):
            raise TypeError(
                "The GDN cold-state mask is not a tensor. Pass a tensor or None."
            )
        return self.state_cache_layout, page_ids, cold

    @staticmethod
    def unpack_paged_state(
        pages: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert cache page axes to the GDN module's state axes."""
        if len(pages) != 2:
            raise ValueError(
                f"The GDN recurrent read returned {len(pages)} states. "
                "Return convolution and recurrent states."
            )
        conv_page, recurrent_page = pages
        return (
            conv_page.transpose(1, 2).contiguous(),
            recurrent_page.transpose(-1, -2).contiguous(),
        )

    def paged_state_update(
        self,
        attn_metadata: dict[str, object],
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[
        MambaStateViewLayout,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Return the cache layout, selected write pages, and packed state."""
        if not self.state_cache_bases:
            raise ValueError(
                "The GDN recurrent cache is not bound. Bind the state cache first."
            )
        page_ids = attn_metadata.get("recurrent_state_write_page_ids")
        if not isinstance(page_ids, torch.Tensor):
            raise TypeError(
                "The GDN write page IDs are not a tensor. Pass recurrent write pages."
            )
        conv_state, recurrent_state = state
        return (
            self.state_cache_layout,
            page_ids,
            (
                conv_state.transpose(-1, -2).contiguous(),
                recurrent_state.transpose(-1, -2).contiguous(),
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        token_valid_mask: torch.Tensor | None = None,
        input_norm: nn.Module | None = None,
        return_state: bool = False,
        return_state_checkpoints: bool = False,
        checkpoint_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the GDN mixer and optionally return its next state."""
        batch, seq_len, _ = hidden_states.shape
        if token_valid_mask is not None and token_valid_mask.shape != (
            batch,
            seq_len,
        ):
            raise ValueError(
                "The GDN token mask has the wrong shape. Use shape [batch, sequence]."
            )
        if (return_state_checkpoints or checkpoint_offsets is not None) and not (
            return_state
        ):
            raise ValueError(
                "The GDN checkpoint request does not return state. "
                "Set return_state=True."
            )
        if return_state_checkpoints and checkpoint_offsets is not None:
            raise ValueError(
                "Two GDN checkpoint modes are active. Select one checkpoint mode."
            )

        if initial_state is None:
            conv_state = None
            recurrent_state = None
        else:
            conv_state, recurrent_state = initial_state

        mixed_qkv, z, b, a = self._project_inputs(hidden_states, input_norm)
        mixed_qkv, next_conv_state = self._apply_conv_with_state(
            mixed_qkv.transpose(1, 2),
            conv_state,
            token_valid_mask,
            return_state_checkpoints=return_state_checkpoints,
            checkpoint_offsets=checkpoint_offsets,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim_per_rank, self.key_dim_per_rank, self.value_dim_per_rank],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch, seq_len, -1, self.head_v_dim)

        is_decode = seq_len == 1
        op = "GDN delta rule decode" if is_decode else "GDN delta rule prefill"
        shape = (
            f"B={batch} S={seq_len} Hk={query.shape[2]} Hv={value.shape[2]} "
            f"Dk={self.head_k_dim} Dv={self.head_v_dim}"
        )
        if return_state_checkpoints:
            reject_reason = "the call requests recurrent checkpoints"
        elif is_decode:
            reject_reason = gdn_decode_reject_reason(query, value, self.norm.eps)
        else:
            reject_reason = gdn_prefill_reject_reason(query, value)

        output_is_normed = False
        if is_decode and reject_reason is None:
            _log_kernel_path(op, "NKI gdn_tkg", shape)
            if recurrent_state is None:
                recurrent_state = query.new_zeros(
                    batch,
                    self.num_v_heads_per_rank,
                    self.head_k_dim,
                    self.head_v_dim,
                    dtype=torch.float32,
                )
            core_out, new_state = gdn_decode(
                query,
                key,
                value,
                a=a,
                b=b,
                z=z,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                norm_weight=self.norm.weight,
                state=recurrent_state,
                token_valid_mask=token_valid_mask,
            )
            output_is_normed = True
        else:
            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
            if token_valid_mask is not None:
                valid = token_valid_mask.unsqueeze(-1)
                beta = torch.where(valid, beta, torch.zeros_like(beta))
                g = torch.where(valid, g, torch.zeros_like(g))

            if not is_decode and reject_reason is None:
                path = (
                    "NKI gdn_cte with checkpoint"
                    if checkpoint_offsets is not None
                    else "NKI gdn_cte"
                )
                _log_kernel_path(op, path, shape)
                core_out, new_state = gdn_prefill(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    mask_pack=self.gdn_mask_pack,
                    checkpoint_offsets=checkpoint_offsets,
                )
            else:
                _log_kernel_path(op, "torch", f"{shape}; {reject_reason}")
                if self.num_v_per_k > 1:
                    query = query.repeat_interleave(self.num_v_per_k, dim=2)
                    key = key.repeat_interleave(self.num_v_per_k, dim=2)
                if return_state_checkpoints:
                    core_out, new_state = recurrent_gated_delta_rule(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=recurrent_state,
                        return_state_checkpoints=True,
                    )
                elif checkpoint_offsets is not None:
                    core_out, new_state = chunked_gated_delta_rule(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=recurrent_state,
                        checkpoint_offsets=checkpoint_offsets,
                    )
                else:
                    delta_rule = (
                        recurrent_gated_delta_rule
                        if is_decode
                        else chunked_gated_delta_rule
                    )
                    core_out, new_state = delta_rule(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=recurrent_state,
                    )

        if not output_is_normed:
            core_out = self.norm(
                core_out.reshape(-1, self.head_v_dim),
                z.reshape(-1, self.head_v_dim),
            )
        core_out = core_out.reshape(batch, seq_len, -1)
        out = self._project_output(core_out)
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        if return_state:
            return out, (next_conv_state, new_state)
        return out


# ── Feed-forward and decoder layer ────────────────────────────────────────────


# The nkilib decode MLP kernel overflows its SBUF stack at I/rank 17408 (TP 1),
# and NF.mlp does not check I. 8704 (TP 2) is the largest size verified to run.
_MLP_KERNEL_MAX_INTER_PER_RANK = 8704


class Qwen3_5MLP(nn.Module):
    """Dense gated feed-forward network. This model has no expert layer.

    The MLP kernel also applies the post-attention RMSNorm of the layer.
    """

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        hidden = config.hidden_size
        inter_per_rank = config.intermediate_size // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(hidden, inter_per_rank, dtype=self.dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(hidden, inter_per_rank, dtype=self.dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(inter_per_rank, hidden, dtype=self.dtype)
        )

        for param in (self.gate_proj_weight, self.up_proj_weight):
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=inter_per_rank,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                ),
            )
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=inter_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, x: torch.Tensor, norm: nn.Module | None = None) -> torch.Tensor:
        """Apply ``norm``, then the MLP. Return the MLP output."""
        fused_norm = _kernel_norm(norm)
        if norm is not None and fused_norm is None:
            # The kernel applies only the RMS norm of this model.
            x, norm = norm(x), None
        hidden = x.reshape(-1, x.shape[-1])
        inter_per_rank = self.gate_proj_weight.shape[1]
        shape = f"T={hidden.shape[0]} H={hidden.shape[1]} I={inter_per_rank}"
        reject_reason = self._kernel_reject_reason(hidden, fused_norm)

        if reject_reason is None:
            norm_args = {}
            if fused_norm is not None:
                norm_args = dict(
                    ln_w=_kernel_gamma(fused_norm, hidden.dtype),
                    norm_type=NormType.RMS_NORM,
                    eps=fused_norm.eps,
                )
            path = "NKI mlp" + (" + norm" if fused_norm else "")
            _log_kernel_path("MLP", path, shape)
            out = NF.mlp(
                hidden,
                self.gate_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
                **norm_args,
            ).reshape(x.shape)
        else:
            _log_kernel_path("MLP", "torch", f"{shape}; {reject_reason}")
            if norm is not None:
                x = norm(x)
            out = F.silu(x @ self.gate_proj_weight) * (x @ self.up_proj_weight)
            out = out @ self.down_proj_weight
        # The down projection is row-parallel. Assign the result; see the note in
        # ``Qwen3_5Attention.forward``.
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out

    def _kernel_reject_reason(
        self, hidden: torch.Tensor, norm: Qwen3_5RMSNorm | None
    ) -> str | None:
        """Return why the MLP kernel cannot run the call, or None if it can.

        Mirror the dispatch of NF.mlp: a fused norm selects the decode kernel
        when that kernel can run the shape.
        """
        inter_per_rank = self.gate_proj_weight.shape[1]
        if not can_run_kernel(hidden):
            return "NKI kernels are disabled or the tensor is on CPU"
        if inter_per_rank > _MLP_KERNEL_MAX_INTER_PER_RANK:
            return (
                f"I/rank {inter_per_rank} is larger than "
                f"{_MLP_KERNEL_MAX_INTER_PER_RANK}, the decode kernel stack limit"
            )
        quantization = QuantizationType.NONE
        if norm is not None and _can_fuse_decode_norm(hidden, quantization):
            mode = ComputationMode.DECODE
        else:
            mode = ComputationMode.AUTO
        if not can_use_mlp_kernel(
            hidden, self.gate_proj_weight, None, None, None, quantization, mode
        ):
            return "the shape is outside the kernel limits"
        return None


class Qwen3_5DecoderLayer(nn.Module):
    """One decoder layer. ``layer_types`` selects the token mixer."""

    reset_dynamo_cache_before_prefill_warmup = True
    attention_cls = Qwen3_5Attention
    gated_delta_net_cls = Qwen3_5GatedDeltaNet
    mlp_cls = Qwen3_5MLP

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == LINEAR_ATTENTION:
            self.linear_attn = self.gated_delta_net_cls(config, layer_idx)
        else:
            self.self_attn = self.attention_cls(config, layer_idx)
        self.mlp = self.mlp_cls(config)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            config.torch_dtype,
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            config.torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        attn_metadata: dict[str, object] | None = None,
        positions: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        *,
        linear_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_linear_state: bool = False,
        return_state_checkpoints: bool = False,
        checkpoint_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one layer and optionally return the next GDN state."""
        residual = hidden_states
        next_state = None
        if self.layer_type == LINEAR_ATTENTION:
            mixed = self.linear_attn(
                hidden_states,
                initial_state=linear_state,
                token_valid_mask=token_valid_mask,
                input_norm=self.input_layernorm,
                return_state=return_linear_state,
                return_state_checkpoints=return_state_checkpoints,
                checkpoint_offsets=checkpoint_offsets,
            )
            if return_linear_state:
                hidden_states, next_state = mixed
            else:
                hidden_states = mixed
        else:
            if return_linear_state:
                raise ValueError(
                    "A full-attention layer has no recurrent state. "
                    "Request state from a GDN layer."
                )
            hidden_states = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask,
                attn_metadata,
                positions,
                input_norm=self.input_layernorm,
            )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(
            hidden_states,
            self.post_attention_layernorm,
        )
        if return_linear_state:
            if next_state is None:
                raise RuntimeError(
                    "The GDN layer returned no state. Return the next GDN state."
                )
            return hidden_states, next_state
        return hidden_states


class Qwen3_5Model(nn.Module):
    """Text backbone of the hybrid model."""

    decoder_layer_cls = Qwen3_5DecoderLayer

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            self.decoder_layer_cls(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            config.torch_dtype,
        )

    def read_recurrent_states(
        self,
        attn_metadata: dict[str, dict[str, object]],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Read every bound GDN state before any decoder layer runs."""
        grouped = {}
        for layer_idx, (layer, layer_type) in enumerate(
            zip(self.layers, self.config.layer_types, strict=True)
        ):
            if layer_type != LINEAR_ATTENTION:
                continue
            mixer = layer.linear_attn
            if not mixer.state_cache_bases:
                continue
            layer_name = f"layers.{layer_idx}.linear_attn"
            if layer_name not in attn_metadata:
                raise KeyError(f"The cache metadata for {layer_name} is missing.")
            group = mixer.state_cache_group
            group_key = group if group is not None else -(layer_idx + 1)
            if group_key not in grouped:
                grouped[group_key] = (mixer.state_cache_bases, [])
            bases, entries = grouped[group_key]
            entries.append((layer_idx, mixer, attn_metadata[layer_name]))

        states = {}
        for bases, entries in grouped.values():
            pages = read_mamba_state_pages_grouped(
                bases,
                [
                    mixer.paged_state_read_request(metadata)
                    for _layer_idx, mixer, metadata in entries
                ],
            )
            for (layer_idx, mixer, _metadata), layer_pages in zip(
                entries,
                pages,
                strict=True,
            ):
                states[layer_idx] = mixer.unpack_paged_state(layer_pages)
        return states

    def write_recurrent_states(
        self,
        attn_metadata: dict[str, dict[str, object]],
        states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Write every GDN state after all decoder layers have run."""
        grouped = {}
        for layer_idx, state in states.items():
            layer = self.layers[layer_idx]
            mixer = layer.linear_attn
            layer_name = f"layers.{layer_idx}.linear_attn"
            if layer_name not in attn_metadata:
                raise KeyError(f"The cache metadata for {layer_name} is missing.")
            group = mixer.state_cache_group
            group_key = group if group is not None else -(layer_idx + 1)
            if group_key not in grouped:
                grouped[group_key] = (mixer.state_cache_bases, [])
            bases, entries = grouped[group_key]
            entries.append(mixer.paged_state_update(attn_metadata[layer_name], state))

        for bases, updates in grouped.values():
            write_mamba_state_pages_grouped(bases, updates)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: dict[str, dict[str, object]] | None = None,
        rotary_position_ids: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        *,
        linear_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        token_valid_mask: torch.Tensor | None = None,
        return_linear_states: bool = False,
        return_state_checkpoints: bool = False,
        checkpoint_offsets: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            dict[int, tuple[torch.Tensor, torch.Tensor]],
        ]
    ):
        """Run the backbone and optionally return each GDN state."""
        if input_ids.ndim == 1:
            if attn_metadata is None:
                raise ValueError(
                    "Flat Qwen3.8 input has no cache metadata. "
                    "Pass metadata to recover the batch shape."
                )
            first_metadata = next(iter(attn_metadata.values()))
            block_table = first_metadata.get("block_table_tensor")
            if not isinstance(block_table, torch.Tensor):
                raise TypeError(
                    "The first cache block table is not a tensor. Pass a tensor."
                )
            batch = block_table.shape[0]
            if input_ids.shape[0] % batch:
                raise ValueError(
                    f"The token count {input_ids.shape[0]} does not divide by "
                    f"the request count {batch}. Pad each request equally."
                )
            seq_len = input_ids.shape[0] // batch
            input_ids = input_ids.reshape(batch, seq_len)
            if rotary_position_ids is not None:
                rotary_position_ids = rotary_position_ids.reshape(3, batch, seq_len)
        elif input_ids.ndim == 2:
            batch, seq_len = input_ids.shape
        else:
            raise ValueError(
                "Qwen3.8 input IDs have the wrong rank. "
                "Use [tokens] or [batch, sequence]."
            )
        if position_ids.numel() != batch * seq_len:
            raise ValueError(
                "Qwen3.8 position IDs do not match the token count. "
                "Provide one position for each token."
            )
        position_ids = position_ids.reshape(batch, seq_len)
        hidden_states = self.embed_tokens(
            input_ids.reshape(-1),
            scatter_tokens=False,
            rank=rank,
        ).view(batch, seq_len, -1)
        rope_positions = (
            position_ids if rotary_position_ids is None else rotary_position_ids
        )
        position_embeddings = self.rotary_emb(rope_positions, hidden_states.dtype)
        attention_mask = build_causal_mask(
            seq_len,
            seq_len,
            hidden_states.dtype,
            hidden_states,
        )

        has_bound_recurrent_cache = any(
            layer_type == LINEAR_ATTENTION and bool(layer.linear_attn.state_cache_bases)
            for layer, layer_type in zip(
                self.layers,
                self.config.layer_types,
                strict=True,
            )
        )
        use_paged_recurrent_cache = (
            has_bound_recurrent_cache and attn_metadata is not None
        )
        if (
            has_bound_recurrent_cache
            and attn_metadata is None
            and linear_states is None
        ):
            raise ValueError(
                "The bound GDN cache has no metadata. Pass scheduler cache metadata."
            )
        if use_paged_recurrent_cache and linear_states is not None:
            raise ValueError(
                "Both paged and explicit GDN states were provided. "
                "Use one recurrent-state source."
            )
        if use_paged_recurrent_cache:
            if token_valid_mask is None:
                first_recurrent_layer = next(
                    layer_idx
                    for layer_idx, layer_type in enumerate(self.config.layer_types)
                    if layer_type == LINEAR_ATTENTION
                )
                recurrent_metadata = attn_metadata[
                    f"layers.{first_recurrent_layer}.linear_attn"
                ]
                state_write_valid = recurrent_metadata.get(
                    "recurrent_state_write_valid_mask"
                )
                if not isinstance(state_write_valid, torch.Tensor):
                    raise TypeError("The GDN write-valid mask must be a tensor.")
                token_valid_mask = build_recurrent_token_mask(
                    position_ids,
                    state_write_valid,
                )
            linear_states = self.read_recurrent_states(attn_metadata)

        next_states = {}
        linear_states = {} if linear_states is None else linear_states
        collect_linear_states = return_linear_states or use_paged_recurrent_cache
        for layer_idx, (layer, layer_type) in enumerate(
            zip(self.layers, self.config.layer_types, strict=True)
        ):
            mixer_name = (
                "linear_attn" if layer_type == LINEAR_ATTENTION else "self_attn"
            )
            layer_metadata = (
                None
                if attn_metadata is None
                else attn_metadata[f"layers.{layer_idx}.{mixer_name}"]
            )
            if layer_type == LINEAR_ATTENTION and collect_linear_states:
                layer_checkpoint_offsets = checkpoint_offsets
                if layer_metadata is not None:
                    metadata_offsets = layer_metadata.get(
                        "recurrent_state_prefill_checkpoint_offsets"
                    )
                    if metadata_offsets is not None and not isinstance(
                        metadata_offsets,
                        torch.Tensor,
                    ):
                        raise TypeError(
                            "The GDN prefill checkpoint offsets are not a tensor. "
                            "Pass a tensor or None."
                        )
                    if metadata_offsets is not None:
                        if layer_checkpoint_offsets is not None:
                            raise ValueError(
                                "The GDN checkpoint offsets have two sources. "
                                "Pass explicit offsets or cache metadata."
                            )
                        layer_checkpoint_offsets = metadata_offsets
                hidden_states, next_state = layer(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    layer_metadata,
                    position_ids,
                    token_valid_mask,
                    linear_state=linear_states.get(layer_idx),
                    return_linear_state=True,
                    return_state_checkpoints=return_state_checkpoints,
                    checkpoint_offsets=layer_checkpoint_offsets,
                )
                next_states[layer_idx] = next_state
            else:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    layer_metadata,
                    position_ids,
                    token_valid_mask,
                )
        hidden_states = self.norm(hidden_states)
        if use_paged_recurrent_cache:
            self.write_recurrent_states(attn_metadata, next_states)
        if return_linear_states:
            return hidden_states, next_states
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.8 hybrid text model with a language-model head."""

    backbone_cls = Qwen3_5Model

    def __init__(self, config: Qwen3_5TextConfig, neuron_config: NeuronConfig | None):
        super().__init__()
        self.config = config
        self.neuron_config = neuron_config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self._validate_tensor_parallel_degree()
        limit = config.sampling_vocab_size
        if limit is not None and not 0 < limit <= config.vocab_size:
            raise ValueError("sampling_vocab_size must be within the checkpoint vocabulary")
        self.sampling_vocab_size = config.vocab_size if limit is None else limit
        self.suppressed_token_ids = tuple(sorted(set(config.suppressed_token_ids)))

        self.language_model = self.backbone_cls(config)
        self.on_device_sampling_config = (
            neuron_config.on_device_sampling_config if neuron_config else None
        )
        self._gather_logits = bool(
            neuron_config
            and (
                neuron_config.max_logprobs != 0
                or neuron_config.debug_logits_dir is not None
            )
        )
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=self.on_device_sampling_config is None,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
            ),
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )
        self.return_mtp_hidden_states = False

    def enable_mtp_hidden_states(self) -> None:
        """Return the final hidden states for an MTP draft model."""
        self.return_mtp_hidden_states = True

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | Qwen3_5TextConfig,
        neuron_config: NeuronConfig | None,
    ) -> "Qwen3_5ForCausalLM":
        """Create the BF16 model from framework configuration."""
        config = (
            hf_config
            if isinstance(hf_config, Qwen3_5TextConfig)
            else Qwen3_5TextConfig.from_configs(hf_config, neuron_config)
        )
        return cls(config, neuron_config)

    def _validate_tensor_parallel_degree(self) -> None:
        """Check that the tensor-parallel size divides each sharded axis."""
        for name, count in (
            ("num_attention_heads", self.config.num_attention_heads),
            ("linear_num_key_heads", self.config.linear_num_key_heads),
            ("linear_num_value_heads", self.config.linear_num_value_heads),
            ("intermediate_size", self.config.intermediate_size),
            ("vocab_size", self.config.vocab_size),
        ):
            if count % self.world_size:
                raise ValueError(
                    f"{name}={count} does not divide by tensor-parallel size "
                    f"{self.world_size}. Select a compatible size."
                )

    def get_kv_spec(self) -> KVSpec:
        """Return the framework-owned hybrid cache specification."""
        return build_qwen3_5_kv_spec(self.config, self.world_size)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        """Bind framework-owned attention and recurrent cache pages."""
        recurrent_cache_groups = {}
        for layer_idx, layer_type in enumerate(self.config.layer_types):
            layer = self.language_model.layers[layer_idx]
            if layer_type == FULL_ATTENTION:
                layer_name = f"layers.{layer_idx}.self_attn"
                views = kv_caches.get(layer_name)
                if views is None:
                    raise KeyError(f"The cache for {layer_name} is not initialized.")
                if len(views) != 2:
                    raise ValueError(
                        f"The cache for {layer_name} has {len(views)} tensors. "
                        "Provide one key tensor and one value tensor."
                    )
                layer.self_attn.k_cache, layer.self_attn.v_cache = views
                continue

            layer_name = f"layers.{layer_idx}.linear_attn"
            views = kv_caches.get(layer_name)
            if views is None:
                raise KeyError(f"The cache for {layer_name} is not initialized.")
            if len(views) != 2:
                raise ValueError(
                    f"The cache for {layer_name} has {len(views)} tensors. "
                    "Provide one convolution tensor and one recurrent tensor."
                )
            bases, layout = prepare_mamba_state_cache_access(views)
            layer.linear_attn.state_cache_bases = tuple(bases)
            layer.linear_attn.state_cache_layout = layout
            storage_key = tuple(
                (base.device, base.untyped_storage()._cdata) for base in bases
            )
            layer.linear_attn.state_cache_group = recurrent_cache_groups.setdefault(
                storage_key,
                len(recurrent_cache_groups),
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        attn_metadata: dict[str, dict[str, object]] | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        rotary_position_ids: torch.Tensor | None = None,
        return_state_checkpoints: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run the BF16 model and return logits or sampled tokens."""
        del kwargs
        if positions is None:
            if input_ids.ndim != 2:
                raise ValueError(
                    "Flat token input has no positions. Provide explicit positions."
                )
            positions = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
            ).expand(input_ids.shape[0], -1)
        hidden_states = self.language_model(
            input_ids,
            positions,
            attn_metadata,
            rotary_position_ids=rotary_position_ids,
            rank=rank,
            return_state_checkpoints=return_state_checkpoints,
        )
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if sampling_positions is not None:
            hidden_states = torch.index_select(flat_hidden, dim=0, index=sampling_positions)
        logits = self.lm_head(hidden_states)
        if self.sampling_vocab_size < self.config.vocab_size or self.suppressed_token_ids:
            width = logits.shape[-1]
            ids = torch.arange(width, dtype=torch.int32, device=logits.device)
            if width < self.config.vocab_size:
                if rank is None:
                    raise ValueError("A TP rank tensor is required to mask a vocabulary shard")
                ids = ids + rank * width
            allowed = ids < self.sampling_vocab_size
            for banned in self.suppressed_token_ids:
                allowed = allowed & (ids != banned)
            logits = logits.masked_fill(~allowed, -torch.inf)

        if sampling_positions is None or self.on_device_sampling_config is None:
            if self.return_mtp_hidden_states:
                return logits, flat_hidden
            return logits

        sampled_tokens = self.sampler(
            logits,
            sampling_params,
            logit_mask=logit_mask,
            tp_rank=rank,
        )
        gathered_logits = (
            self.tp_group.all_gather(logits, dim=1) if self._gather_logits else None
        )
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            sampled_tokens = rejection_sampler(spec_decode_metadata, sampled_tokens)
            if self.return_mtp_hidden_states:
                return sampled_tokens, flat_hidden, gathered_logits
            return sampled_tokens
        if self.return_mtp_hidden_states:
            return sampled_tokens, flat_hidden, gathered_logits
        return sampled_tokens, gathered_logits

    def build_weight_mappings(self) -> dict[str, str | list[str]]:
        """Return the map from model parameters to checkpoint tensors."""
        return build_weight_mappings(self.config)

    @staticmethod
    def unused_checkpoint_prefixes() -> tuple[str, ...]:
        """Return the checkpoint prefixes that this model does not load."""
        return UNUSED_HF_PREFIXES

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None,
    ) -> None:
        """Load and convert the checkpoint tensors for this rank."""
        mappings = self.build_weight_mappings()
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank,
            self.world_size,
            self,
            mappings,
            device,
        ).state_dict

        # The loader casts each tensor to the dtype of its parameter. Read the
        # stored dtypes, so that an FP8 tensor cannot load into a BF16
        # parameter without its scale.
        stored = checkpoint.tensor_dtypes()
        fp8_keys = sorted(
            key
            for source in mappings.values()
            for key in (source if isinstance(source, list) else [source])
            if stored.get(key, "").startswith("F8")
        )
        if fp8_keys:
            raise ValueError(
                f"The checkpoint stores {len(fp8_keys)} tensors in FP8, for "
                f"example {fp8_keys[0]}. The BF16 model reads BF16 tensors. Use "
                "a BF16 checkpoint, or an FP8 checkpoint with its "
                "quantization_config."
            )

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if name.endswith(("dt_bias", "A_log")):
                rank_sharded[name] = tensor.to(torch.float32)
            elif tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)


def build_weight_mappings(config: Qwen3_5TextConfig) -> dict[str, str | list[str]]:
    """Return the map from each parameter name to its checkpoint tensor names.

    A joined weight maps to the list of tensors that it joins. The vision tower and
    the multi-token-prediction head are not loaded (``UNUSED_HF_PREFIXES``).
    """
    mappings: dict[str, str | list[str]] = {
        "language_model.embed_tokens.weight": f"{HF_TEXT_PREFIX}.embed_tokens.weight",
        "language_model.norm.weight": f"{HF_TEXT_PREFIX}.norm.weight",
        # A checkpoint with tied embeddings has no lm_head tensor. The LM head
        # then reads the embedding, which has the same [vocab, hidden] layout.
        "lm_head.weight": (
            f"{HF_TEXT_PREFIX}.embed_tokens.weight"
            if config.tie_word_embeddings
            else "lm_head.weight"
        ),
    }

    for layer_idx, layer_type in enumerate(config.layer_types):
        param = f"language_model.layers.{layer_idx}"
        hf = f"{HF_TEXT_PREFIX}.layers.{layer_idx}"

        mappings[f"{param}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
        mappings[f"{param}.post_attention_layernorm.weight"] = (
            f"{hf}.post_attention_layernorm.weight"
        )
        for proj in ("gate_proj", "up_proj", "down_proj"):
            mappings[f"{param}.mlp.{proj}_weight"] = f"{hf}.mlp.{proj}.weight"

        if layer_type == FULL_ATTENTION:
            mappings[f"{param}.self_attn.qkv_proj_weight"] = [
                f"{hf}.self_attn.{proj}.weight"
                for proj in ("q_proj", "k_proj", "v_proj")
            ]
            mappings[f"{param}.self_attn.o_proj_weight"] = (
                f"{hf}.self_attn.o_proj.weight"
            )
            for norm in ("q_norm", "k_norm"):
                mappings[f"{param}.self_attn.{norm}.weight"] = (
                    f"{hf}.self_attn.{norm}.weight"
                )
        else:
            mappings[f"{param}.linear_attn.in_proj_qkvz_weight"] = [
                f"{hf}.linear_attn.in_proj_qkv.weight",
                f"{hf}.linear_attn.in_proj_z.weight",
            ]
            for proj in ("in_proj_a", "in_proj_b"):
                mappings[f"{param}.linear_attn.{proj}_weight"] = (
                    f"{hf}.linear_attn.{proj}.weight"
                )
            mappings[f"{param}.linear_attn.conv1d_weight"] = (
                f"{hf}.linear_attn.conv1d.weight"
            )
            mappings[f"{param}.linear_attn.out_proj_weight"] = (
                f"{hf}.linear_attn.out_proj.weight"
            )
            mappings[f"{param}.linear_attn.norm.weight"] = (
                f"{hf}.linear_attn.norm.weight"
            )
            mappings[f"{param}.linear_attn.dt_bias"] = f"{hf}.linear_attn.dt_bias"
            mappings[f"{param}.linear_attn.A_log"] = f"{hf}.linear_attn.A_log"

    return mappings
