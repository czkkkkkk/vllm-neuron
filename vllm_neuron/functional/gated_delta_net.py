# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic Gated DeltaNet math and NKI kernel wrappers."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.experimental.gdn.gdn_conv1d import (
    gdn_conv1d_decode as nkilib_gdn_conv1d_decode,
)
from nkilib.experimental.gdn.gdn_conv1d import (
    gdn_conv1d_prefill as nkilib_gdn_conv1d_prefill,
)
from nkilib.experimental.gdn.gdn_cte import gdn_cte as nkilib_gdn_prefill
from nkilib.experimental.gdn.gdn_cte_utils import CHUNK, MAX_HEAD_DIM, build_mask_pack
from nkilib.experimental.gdn.gdn_tkg import gdn_tkg as nkilib_gdn_decode
from torch import Tensor

from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

wrapped_gdn_conv1d_prefill = wrap_nki(nkilib_gdn_conv1d_prefill)
wrapped_gdn_conv1d_decode = wrap_nki(nkilib_gdn_conv1d_decode)
wrapped_gdn_prefill = wrap_nki(nkilib_gdn_prefill)
wrapped_gdn_decode = wrap_nki(nkilib_gdn_decode)

# The prefill convolution takes the sequence plus the K - 1 window tokens of the
# conv state. For each channel tile it keeps two fp32 [128, T] tiles and one bf16
# [128, T] tile, 10 * T bytes in each partition. The limit serves the 8192-token
# prefill bucket at K = 4 and uses 82 KB of the 256 KiB partition on Trn3.
_CONV_PREFILL_MAX_TOKENS = 8192 + 4 - 1
_GDN_DECODE_EPS = 1e-6
_NO_OP_GATE_INPUT = -1.0e4


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """Normalize a tensor with the L2 norm.

    Example:
        >>> l2norm(torch.tensor([[3.0, 4.0]])).round(decimals=2).tolist()
        [[0.6000000238418579, 0.800000011920929]]
    """
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    return_state_checkpoints: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run the gated delta rule one token at a time.

    Example:
        >>> output, state = recurrent_gated_delta_rule(  # doctest: +SKIP
        ...     query, key, value, decay, beta
        ... )
    """
    in_dtype = query.dtype
    batch, seq_len, num_v_heads, head_k_dim = key.shape
    head_v_dim = value.shape[-1]

    query, key, value, beta, g = (
        x.transpose(1, 2).to(torch.float32) for x in (query, key, value, beta, g)
    )
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query = query / (head_k_dim**0.5)

    if initial_state is None:
        state = torch.zeros(
            (batch, num_v_heads, head_k_dim, head_v_dim),
            dtype=torch.float32,
            device=value.device,
        )
    else:
        state = initial_state.to(torch.float32)

    outputs = []
    state_checkpoints = []
    for token_index in range(seq_len):
        query_token = query[:, :, token_index]
        key_token = key[:, :, token_index]
        value_token = value[:, :, token_index]
        state = state * g[:, :, token_index].exp()[..., None, None]
        predicted = (state * key_token.unsqueeze(-1)).sum(dim=-2)
        delta = (value_token - predicted) * beta[:, :, token_index].unsqueeze(-1)
        state = state + key_token.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * query_token.unsqueeze(-1)).sum(dim=-2))
        if return_state_checkpoints:
            state_checkpoints.append(state)

    out = torch.stack(outputs, dim=2).transpose(1, 2).to(in_dtype)
    next_state = (
        torch.stack(state_checkpoints, dim=1) if return_state_checkpoints else state
    )
    return out, next_state


def unit_lower_triangular_inverse(a: Tensor, chunk_size: int) -> Tensor:
    """Return ``inv(I - a)`` for a strictly lower triangular matrix.

    Example:
        >>> matrix = torch.zeros(1, 2, 2)
        >>> unit_lower_triangular_inverse(matrix, 2).tolist()
        [[[1.0, 0.0], [0.0, 1.0]]]
    """
    eye = torch.eye(chunk_size, dtype=a.dtype, device=a.device)
    inverse = eye + a
    power = a
    steps = max(chunk_size - 1, 1).bit_length()
    for _ in range(steps - 1):
        power = power @ power
        inverse = inverse @ (eye + power)
    return inverse


def chunked_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    chunk_size: int = 16,
    checkpoint_offsets: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the gated delta rule in sequence chunks.

    Example:
        >>> output, state = chunked_gated_delta_rule(  # doctest: +SKIP
        ...     query, key, value, decay, beta
        ... )
    """
    in_dtype = query.dtype
    batch, seq_len, num_v_heads, head_k_dim = key.shape
    head_v_dim = value.shape[-1]

    checkpoint_state = None
    if checkpoint_offsets is not None:
        checkpoint_offsets = checkpoint_offsets.reshape(-1).to(
            device=g.device,
            dtype=torch.int64,
        )
        if checkpoint_offsets.shape != (batch,):
            raise ValueError(
                f"The checkpoint list has {checkpoint_offsets.numel()} values, "
                f"but the batch has {batch} requests. Provide one value for "
                "each request."
            )
        # Zero gates after the offset keep the state unchanged, so a second
        # chunked pass returns the checkpoint state. The work grows with the
        # chunk count, not with the token count.
        positions = torch.arange(seq_len, device=g.device)
        keep = (positions.unsqueeze(0) < checkpoint_offsets.unsqueeze(1)).unsqueeze(-1)
        _, checkpoint_state = chunked_gated_delta_rule(
            query,
            key,
            value,
            torch.where(keep, g, torch.zeros_like(g)),
            torch.where(keep, beta, torch.zeros_like(beta)),
            initial_state=initial_state,
            chunk_size=chunk_size,
        )

    query, key, value, beta, decay = (
        x.transpose(1, 2).to(torch.float32) for x in (query, key, value, beta, g)
    )
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query = query * (head_k_dim**-0.5)

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
        beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    total_len = seq_len + pad
    num_chunks = total_len // chunk_size

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)

    def to_chunks(x: Tensor) -> Tensor:
        return x.reshape(
            x.shape[0],
            x.shape[1],
            num_chunks,
            chunk_size,
            x.shape[-1],
        )

    query, key, key_beta, value_beta = (
        to_chunks(x) for x in (query, key, key_beta, value_beta)
    )
    decay = decay.reshape(decay.shape[0], decay.shape[1], num_chunks, chunk_size)

    cumulative_decay = decay.cumsum(dim=3)
    strictly_upper = torch.ones(
        chunk_size,
        chunk_size,
        dtype=torch.bool,
        device=query.device,
    ).triu(1)
    pairwise_decay = cumulative_decay.unsqueeze(4) - cumulative_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(
        strictly_upper,
        float("-inf"),
    ).exp()

    attention_system = (key_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attention = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_key_beta = key_beta * cumulative_decay.exp().unsqueeze(-1)

    inverse = unit_lower_triangular_inverse(
        -attention_system.tril(-1),
        chunk_size,
    )
    new_values = inverse @ value_beta
    key_cumulative_decay = inverse @ decayed_key_beta

    if initial_state is None:
        state = torch.zeros(
            (batch, num_v_heads, head_k_dim, head_v_dim),
            dtype=torch.float32,
            device=value.device,
        )
    else:
        state = initial_state.to(torch.float32)
    query = query * cumulative_decay.exp().unsqueeze(-1)
    key = key * (cumulative_decay[..., -1:] - cumulative_decay).exp().unsqueeze(-1)
    chunk_decay = cumulative_decay[..., -1].exp()[..., None, None]

    outputs = []
    for chunk_index in range(num_chunks):
        new_value = (
            new_values[:, :, chunk_index]
            - key_cumulative_decay[:, :, chunk_index] @ state
        )
        inter_chunk_attention = query[:, :, chunk_index] @ state
        outputs.append(
            inter_chunk_attention + intra_chunk_attention[:, :, chunk_index] @ new_value
        )
        state = (
            state * chunk_decay[:, :, chunk_index]
            + key[:, :, chunk_index].transpose(-1, -2) @ new_value
        )

    out = torch.stack(outputs, dim=2).reshape(
        batch,
        num_v_heads,
        total_len,
        head_v_dim,
    )
    out = out[:, :, :seq_len].transpose(1, 2).to(in_dtype)
    if checkpoint_state is not None:
        state = torch.stack((checkpoint_state, state), dim=1)
    return out, state


def build_gdn_mask_pack() -> Tensor:
    """Return the constant CPU mask pack for ``gdn_prefill``.

    Example:
        >>> mask_pack = build_gdn_mask_pack()
        >>> mask_pack.device.type
        'cpu'
    """
    return torch.tensor(
        np.ascontiguousarray(build_mask_pack()),
        dtype=torch.float32,
        device="cpu",
    )


def gdn_conv1d_reject_reason(
    mixed: Tensor,
    conv_state: Tensor | None,
    kernel_size: int,
) -> str | None:
    """Return the reason that the NKI convolution cannot run.

    Example:
        >>> gdn_conv1d_reject_reason(  # doctest: +SKIP
        ...     mixed, conv_state, kernel_size=4
        ... )
    """
    if not can_run_kernel(mixed):
        return "NKI kernels are disabled or the tensor is on CPU"
    if mixed.dtype != torch.bfloat16:
        return f"dtype {mixed.dtype} is not bfloat16"
    if conv_state is not None and conv_state.shape[-1] != kernel_size - 1:
        return f"conv_state holds {conv_state.shape[-1]} tokens, not {kernel_size - 1}"
    seq_len = mixed.shape[-1]
    if seq_len > 1 and seq_len + kernel_size - 1 > _CONV_PREFILL_MAX_TOKENS:
        return (
            f"seq {seq_len} plus the window is larger than "
            f"{_CONV_PREFILL_MAX_TOKENS} tokens"
        )
    return None


def gdn_conv1d(
    mixed: Tensor,
    conv_weight: Tensor,
    conv_state: Tensor | None,
) -> Tensor:
    """Run the depthwise causal convolution and SiLU.

    Example:
        >>> output = gdn_conv1d(  # doctest: +SKIP
        ...     mixed, conv_weight, conv_state
        ... )
    """
    batch, conv_dim, seq_len = mixed.shape
    if conv_weight.dim() != 2 or conv_weight.shape[0] != conv_dim:
        raise ValueError(
            f"The convolution weight has shape {tuple(conv_weight.shape)}. "
            f"Use a [channels, kernel] weight with {conv_dim} channels, for "
            "example by squeezing the depthwise axis of a [C, 1, K] weight."
        )
    window = conv_weight.shape[-1] - 1
    if conv_state is None:
        conv_state = mixed.new_zeros(batch, conv_dim, window)
    conv_state = conv_state.to(mixed.dtype)
    conv_weight = conv_weight.contiguous()

    if seq_len == 1:
        full_window = torch.cat(
            [conv_state.new_zeros(batch, conv_dim, 1), conv_state],
            -1,
        )
        out, _ = wrapped_gdn_conv1d_decode(
            x=mixed.contiguous(),
            conv_state=full_window.contiguous(),
            conv_weight=conv_weight,
        )
        return out

    padded = torch.cat([conv_state, mixed], dim=-1).contiguous()
    out, _ = wrapped_gdn_conv1d_prefill(
        mixed=padded,
        conv_weight=conv_weight,
    )
    return out[:, :, window:]


def gdn_prefill_reject_reason(
    query: Tensor,
    value: Tensor,
) -> str | None:
    """Return the reason that the NKI prefill delta rule cannot run.

    Example:
        >>> gdn_prefill_reject_reason(query, value)  # doctest: +SKIP
    """
    if not can_run_kernel(query):
        return "NKI kernels are disabled or the tensor is on CPU"
    if query.dtype != torch.bfloat16:
        return f"dtype {query.dtype} is not bfloat16"
    _, seq_len, num_k_heads, head_k_dim = query.shape
    num_v_heads, head_v_dim = value.shape[2], value.shape[3]
    if seq_len % CHUNK:
        return f"seq {seq_len} is not a multiple of {CHUNK}"
    if head_k_dim > MAX_HEAD_DIM or head_v_dim > MAX_HEAD_DIM:
        return f"head dims {head_k_dim}/{head_v_dim} are larger than {MAX_HEAD_DIM}"
    if num_v_heads % num_k_heads:
        return f"{num_v_heads} value heads do not divide by {num_k_heads} key heads"
    return None


def gdn_prefill(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None,
    mask_pack: Tensor,
    checkpoint_offsets: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the chunked gated delta rule with the NKI prefill kernel.

    Example:
        >>> output, state = gdn_prefill(  # doctest: +SKIP
        ...     query, key, value, decay, beta, state, mask_pack
        ... )
    """
    head_k_dim = key.shape[-1]
    dtype = query.dtype

    def l2norm_heads(x: Tensor) -> Tensor:
        return l2norm(x.float()).to(dtype).transpose(1, 2).contiguous()

    normalized_query = l2norm_heads(query)
    normalized_key = l2norm_heads(key)
    transposed_value = value.transpose(1, 2).contiguous()

    def run_kernel(kernel_g: Tensor, kernel_beta: Tensor) -> tuple[Tensor, Tensor]:
        state = None
        if initial_state is not None:
            state = initial_state.to(torch.float32).clone()
        return wrapped_gdn_prefill[get_logical_nc_config()](
            query=normalized_query,
            key=normalized_key,
            value=transposed_value,
            g_log=kernel_g.float().transpose(1, 2).contiguous(),
            beta=kernel_beta.float().transpose(1, 2).contiguous(),
            scale=1.0 / math.sqrt(head_k_dim),
            mask_pack=mask_pack,
            recurrent_state=state,
        )

    out, new_state = run_kernel(g, beta)
    if checkpoint_offsets is not None:
        checkpoint_offsets = checkpoint_offsets.reshape(-1).to(
            device=query.device,
            dtype=torch.int64,
        )
        if checkpoint_offsets.shape != (query.shape[0],):
            raise ValueError(
                f"The checkpoint list has {checkpoint_offsets.numel()} values, "
                f"but the batch has {query.shape[0]} requests. Provide one "
                "value for each request."
            )
        positions = torch.arange(query.shape[1], device=query.device)
        checkpoint_mask = positions.unsqueeze(0) < checkpoint_offsets.unsqueeze(1)
        checkpoint_mask = checkpoint_mask.unsqueeze(-1)
        checkpoint_g = torch.where(
            checkpoint_mask,
            g,
            torch.zeros_like(g),
        )
        checkpoint_beta = torch.where(
            checkpoint_mask,
            beta,
            torch.zeros_like(beta),
        )
        # TODO: Return the checkpoint state from the first kernel call. The
        # second call repeats the GDN prefill work for APC.
        _, checkpoint_state = run_kernel(checkpoint_g, checkpoint_beta)
        new_state = torch.stack((checkpoint_state, new_state), dim=1)
    return out.transpose(1, 2), new_state


def gdn_decode_reject_reason(
    query: Tensor,
    value: Tensor,
    norm_eps: float,
) -> str | None:
    """Return the reason that the NKI decode step cannot run.

    Example:
        >>> gdn_decode_reject_reason(  # doctest: +SKIP
        ...     query, value, norm_eps=1e-6
        ... )
    """
    if not can_run_kernel(query):
        return "NKI kernels are disabled or the tensor is on CPU"
    if query.dtype != torch.bfloat16:
        return f"dtype {query.dtype} is not bfloat16"
    _, seq_len, num_k_heads, head_k_dim = query.shape
    num_v_heads, head_v_dim = value.shape[2], value.shape[3]
    if seq_len != 1:
        return f"seq {seq_len} is not one token"
    if head_k_dim > MAX_HEAD_DIM or head_v_dim > MAX_HEAD_DIM:
        return f"head dims {head_k_dim}/{head_v_dim} are larger than {MAX_HEAD_DIM}"
    if num_v_heads % num_k_heads:
        return f"{num_v_heads} value heads do not divide by {num_k_heads} key heads"
    if norm_eps != _GDN_DECODE_EPS:
        return f"norm eps {norm_eps} is not the kernel eps {_GDN_DECODE_EPS}"
    return None


def mask_gdn_decode_gates(
    a: Tensor,
    b: Tensor,
    token_valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Make the decay and update gates no-ops for invalid tokens.

    Example:
        >>> masked_a, masked_b = mask_gdn_decode_gates(  # doctest: +SKIP
        ...     a, b, token_valid_mask
        ... )
    """
    if token_valid_mask is None:
        return a, b
    batch = a.shape[0]
    if token_valid_mask.numel() != batch:
        raise ValueError(
            f"The token mask has {token_valid_mask.numel()} values, but the "
            f"batch has {batch} requests. Provide one value for each request."
        )
    # Broadcast one value per request over the remaining gate axes.
    valid = token_valid_mask.reshape(batch, *([1] * (a.dim() - 1))).to(torch.bool)
    return (
        torch.where(valid, a, torch.full_like(a, _NO_OP_GATE_INPUT)),
        torch.where(valid, b, torch.full_like(b, _NO_OP_GATE_INPUT)),
    )


def repeat_gdn_key_heads(
    query: Tensor,
    key: Tensor,
    num_v_heads: int,
) -> tuple[Tensor, Tensor]:
    """Repeat each key head for its value heads.

    Example:
        >>> repeated_query, repeated_key = repeat_gdn_key_heads(  # doctest: +SKIP
        ...     query, key, num_v_heads=4
        ... )
    """
    num_k_heads = query.shape[2]
    if key.shape[2] != num_k_heads:
        raise ValueError(
            f"The query has {num_k_heads} key heads, but the key has "
            f"{key.shape[2]} heads. Use the same head count."
        )
    if num_v_heads % num_k_heads:
        raise ValueError(
            f"The {num_v_heads} value heads do not divide by the "
            f"{num_k_heads} key heads. Use a divisible head count."
        )
    repeat = num_v_heads // num_k_heads
    return (
        query.repeat_interleave(repeat, dim=2),
        key.repeat_interleave(repeat, dim=2),
    )


def gdn_decode(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    a: Tensor,
    b: Tensor,
    z: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    state: Tensor,
    token_valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run one gated delta rule step with the NKI decode kernel.

    Example:
        >>> output, state = gdn_decode(  # doctest: +SKIP
        ...     query, key, value, a, b, z, A_log, dt_bias,
        ...     norm_weight, state
        ... )
    """
    batch, _, _, head_k_dim = query.shape
    num_v_heads, head_v_dim = value.shape[2], value.shape[3]
    rows = batch * num_v_heads
    a, b = mask_gdn_decode_gates(a, b, token_valid_mask)
    query, key = repeat_gdn_key_heads(query, key, num_v_heads)

    out, new_state = wrapped_gdn_decode[min(get_logical_nc_config(), rows)](
        q=query.reshape(rows, head_k_dim).contiguous(),
        k=key.reshape(rows, head_k_dim).contiguous(),
        v=value.reshape(rows, head_v_dim).contiguous(),
        b=b.reshape(rows).contiguous(),
        a=a.reshape(rows).contiguous(),
        A_log=A_log.float().contiguous(),
        dt_bias=dt_bias.float().contiguous(),
        z=z.reshape(rows, head_v_dim).contiguous(),
        norm_weight=norm_weight.contiguous(),
        state_in=state.to(torch.float32)
        .reshape(rows, head_k_dim, head_v_dim)
        .contiguous(),
    )
    out = out.reshape(batch, 1, num_v_heads * head_v_dim)
    new_state = new_state.reshape(
        batch,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    return out, new_state
