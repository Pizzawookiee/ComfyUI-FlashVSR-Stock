"""Native compact-cache preparation for SpargeAttn lower CUDA ABIs.

These original Triton adapters consume FlashVSR's row-INT8 carriers directly.
They deliberately do not copy or modify SpargeAttn kernels; the resulting
Q/K/V tensors are passed to the compatible symbols exposed by its wheel.
"""

from __future__ import annotations

import math

import torch


class NativeSpargeUnavailable(RuntimeError):
    """Expected capability/API miss selecting the v0.33 compatibility path."""


def _triton_modules():
    try:
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as error:
        raise NativeSpargeUnavailable(
            "Triton is unavailable for native compact-cache conversion"
        ) from error
    return triton, tl


triton, tl = _triton_modules()


@triton.jit
def _row_int8_k_kernel(
    qdata, row_scale, token_map, reference, output, output_scale,
    SOURCE_TOKENS: tl.constexpr, TOTAL_TOKENS: tl.constexpr,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr, OUTPUT_OFFSET: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    rows = block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, HEAD_DIM)
    valid = rows < SOURCE_TOKENS
    source_rows = tl.load(token_map + rows, mask=valid, other=0)
    source_base = (
        ((batch * SOURCE_TOKENS + source_rows) * HEADS + head)
        * HEAD_DIM
    )
    values = tl.load(
        qdata + source_base[:, None] + dims[None, :],
        mask=valid[:, None], other=0,
    ).to(tl.float32)
    scales = tl.load(
        row_scale
        + (batch * SOURCE_TOKENS + source_rows) * HEADS + head,
        mask=valid, other=0.0,
    )
    mean = tl.load(
        reference + (batch * HEADS + head) * HEAD_DIM + dims
    )
    values = values * scales[:, None] - mean[None, :]
    magnitude = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    quant_scale = tl.maximum(magnitude / 127.0, 1.0e-8)
    scaled = values / quant_scale
    scaled += 0.5 * tl.where(scaled >= 0.0, 1.0, -1.0)
    quantized = tl.maximum(-127.0, tl.minimum(127.0, scaled))
    output_rows = OUTPUT_OFFSET + rows
    output_base = (
        ((batch * HEADS + head) * TOTAL_TOKENS + output_rows)
        * HEAD_DIM
    )
    tl.store(
        output + output_base[:, None] + dims[None, :],
        quantized.to(tl.int8), mask=valid[:, None],
    )
    blocks_total = TOTAL_TOKENS // BLOCK_K
    tl.store(
        output_scale
        + (batch * HEADS + head) * blocks_total
        + OUTPUT_OFFSET // BLOCK_K + block,
        quant_scale,
    )


@triton.jit
def _float_k_kernel(
    source, token_map, reference, output, output_scale,
    SOURCE_TOKENS: tl.constexpr, TOTAL_TOKENS: tl.constexpr,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr, OUTPUT_OFFSET: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    rows = block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, HEAD_DIM)
    valid = rows < SOURCE_TOKENS
    source_rows = tl.load(token_map + rows, mask=valid, other=0)
    source_base = (
        (batch * SOURCE_TOKENS + source_rows)
        * (HEADS * HEAD_DIM) + head * HEAD_DIM
    )
    values = tl.load(
        source + source_base[:, None] + dims[None, :],
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)
    mean = tl.load(
        reference + (batch * HEADS + head) * HEAD_DIM + dims
    )
    values -= mean[None, :]
    magnitude = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    quant_scale = tl.maximum(magnitude / 127.0, 1.0e-8)
    scaled = values / quant_scale
    scaled += 0.5 * tl.where(scaled >= 0.0, 1.0, -1.0)
    quantized = tl.maximum(-127.0, tl.minimum(127.0, scaled))
    output_rows = OUTPUT_OFFSET + rows
    output_base = (
        ((batch * HEADS + head) * TOTAL_TOKENS + output_rows)
        * HEAD_DIM
    )
    tl.store(
        output + output_base[:, None] + dims[None, :],
        quantized.to(tl.int8), mask=valid[:, None],
    )
    blocks_total = TOTAL_TOKENS // BLOCK_K
    tl.store(
        output_scale
        + (batch * HEADS + head) * blocks_total
        + OUTPUT_OFFSET // BLOCK_K + block,
        quant_scale,
    )


@triton.jit
def _row_int8_v_amax_kernel(
    qdata, row_scale, token_map, partial,
    SOURCE_TOKENS: tl.constexpr, HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    dim = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    maximum = 0.0
    for block in range(0, tl.cdiv(SOURCE_TOKENS, BLOCK_N)):
        rows = block * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = rows < SOURCE_TOKENS
        source_rows = tl.load(token_map + rows, mask=valid, other=0)
        offsets = (
            ((batch * SOURCE_TOKENS + source_rows) * HEADS + head)
            * HEAD_DIM + dim
        )
        values = tl.load(qdata + offsets, mask=valid, other=0).to(tl.float32)
        scales = tl.load(
            row_scale
            + (batch * SOURCE_TOKENS + source_rows) * HEADS + head,
            mask=valid, other=0.0,
        )
        maximum = tl.maximum(maximum, tl.max(tl.abs(values * scales), axis=0))
    tl.store(partial + (batch * HEADS + head) * HEAD_DIM + dim, maximum)


@triton.jit
def _float_v_amax_kernel(
    source, token_map, partial,
    SOURCE_TOKENS: tl.constexpr, HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    dim = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    maximum = 0.0
    for block in range(0, tl.cdiv(SOURCE_TOKENS, BLOCK_N)):
        rows = block * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = rows < SOURCE_TOKENS
        source_rows = tl.load(token_map + rows, mask=valid, other=0)
        offsets = (
            (batch * SOURCE_TOKENS + source_rows)
            * (HEADS * HEAD_DIM) + head * HEAD_DIM + dim
        )
        values = tl.load(source + offsets, mask=valid, other=0.0)
        maximum = tl.maximum(maximum, tl.max(tl.abs(values), axis=0))
    tl.store(partial + (batch * HEADS + head) * HEAD_DIM + dim, maximum)


@triton.jit
def _row_int8_v_fp8_kernel(
    qdata, row_scale, token_map, scale, output,
    SOURCE_TOKENS: tl.constexpr, PADDED_TOTAL: tl.constexpr,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    OUTPUT_OFFSET: tl.constexpr, SCALE_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    rows = block * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    valid = rows < SOURCE_TOKENS
    source_rows = tl.load(token_map + rows, mask=valid, other=0)
    source_base = (
        ((batch * SOURCE_TOKENS + source_rows) * HEADS + head)
        * HEAD_DIM
    )
    values = tl.load(
        qdata + source_base[:, None] + dims[None, :],
        mask=valid[:, None], other=0,
    ).to(tl.float32)
    row_scales = tl.load(
        row_scale
        + (batch * SOURCE_TOKENS + source_rows) * HEADS + head,
        mask=valid, other=0.0,
    )
    v_scale = tl.load(
        scale + (batch * HEADS + head) * HEAD_DIM + dims
    )
    values = values * row_scales[:, None] / tl.maximum(
        v_scale[None, :], 1.0e-8
    )
    values = tl.maximum(-SCALE_MAX, tl.minimum(SCALE_MAX, values))
    within = rows % 16
    permuted = (
        (rows // 16) * 16
        + (within // 8) * 2
        + ((within // 2) % 4) * 4
        + within % 2
    )
    output_rows = OUTPUT_OFFSET + permuted
    output_base = (
        ((batch * HEADS + head) * HEAD_DIM + dims[None, :])
        * PADDED_TOTAL + output_rows[:, None]
    )
    tl.store(output + output_base, values, mask=valid[:, None])


@triton.jit
def _float_v_fp8_kernel(
    source, token_map, scale, output,
    SOURCE_TOKENS: tl.constexpr, PADDED_TOTAL: tl.constexpr,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    OUTPUT_OFFSET: tl.constexpr, SCALE_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    rows = block * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    valid = rows < SOURCE_TOKENS
    source_rows = tl.load(token_map + rows, mask=valid, other=0)
    source_base = (
        (batch * SOURCE_TOKENS + source_rows)
        * (HEADS * HEAD_DIM) + head * HEAD_DIM
    )
    values = tl.load(
        source + source_base[:, None] + dims[None, :],
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)
    v_scale = tl.load(
        scale + (batch * HEADS + head) * HEAD_DIM + dims
    )
    values /= tl.maximum(v_scale[None, :], 1.0e-8)
    values = tl.maximum(-SCALE_MAX, tl.minimum(SCALE_MAX, values))
    within = rows % 16
    permuted = (
        (rows // 16) * 16
        + (within // 8) * 2
        + ((within // 2) % 4) * 4
        + within % 2
    )
    output_rows = OUTPUT_OFFSET + permuted
    output_base = (
        ((batch * HEADS + head) * HEAD_DIM + dims[None, :])
        * PADDED_TOTAL + output_rows[:, None]
    )
    tl.store(output + output_base, values, mask=valid[:, None])


def native_k(
    descriptor, acquired, block_indices, block_k, profiler=None,
):
    current = descriptor.current_k
    batch, current_tokens, channels = current.shape
    heads = descriptor.heads
    head_dim = channels // heads
    slot_tokens = descriptor.tokens_per_frame * 2
    total_tokens = slot_tokens * len(acquired) + current_tokens
    if total_tokens % block_k:
        raise NativeSpargeUnavailable("K length is not block aligned")
    output = torch.empty(
        (batch, heads, total_tokens, head_dim),
        device=current.device, dtype=torch.int8,
    )
    scales = torch.empty(
        (batch, heads, total_tokens // block_k),
        device=current.device, dtype=torch.float32,
    )
    reference = descriptor.k_reference_mean.to(
        device=current.device, dtype=torch.float32
    ).contiguous()
    total_marker = profiler.profile_start(current) if profiler else None
    marker = profiler.profile_start(current) if profiler else None
    offset = 0
    grid = (triton.cdiv(slot_tokens, block_k), heads, batch)
    for cached_k, _cached_v in acquired:
        _row_int8_k_kernel[grid](
            cached_k.qdata, cached_k.scale, block_indices,
            reference, output, scales,
            SOURCE_TOKENS=slot_tokens, TOTAL_TOKENS=total_tokens,
            HEADS=heads, HEAD_DIM=head_dim, BLOCK_K=block_k,
            OUTPUT_OFFSET=offset,
        )
        offset += slot_tokens
    grid = (triton.cdiv(current_tokens, block_k), heads, batch)
    _float_k_kernel[grid](
        current, block_indices, reference, output, scales,
        SOURCE_TOKENS=current_tokens, TOTAL_TOKENS=total_tokens,
        HEADS=heads, HEAD_DIM=head_dim, BLOCK_K=block_k,
        OUTPUT_OFFSET=offset,
    )
    if profiler:
        profiler.profile_end("sparge_k_int8_to_block_int8", marker)
        profiler.profile_end("sparge_k_native_total", total_marker)
    return output, scales


def native_v_fp8(
    descriptor, acquired, block_indices, arch, profiler=None,
):
    current = descriptor.current_v
    batch, current_tokens, channels = current.shape
    heads = descriptor.heads
    head_dim = channels // heads
    slot_tokens = descriptor.tokens_per_frame * 2
    total_tokens = slot_tokens * len(acquired) + current_tokens
    padded_total = math.ceil(total_tokens / 128) * 128
    scale_max = 448.0 if arch == "sm90" else 2.25

    scale_marker = profiler.profile_start(current) if profiler else None
    source_count = len(acquired) + 1
    partials = torch.empty(
        (source_count, batch, heads, head_dim),
        device=current.device,
        dtype=torch.float32,
    )
    grid = (head_dim, heads, batch)
    for source_index, (_cached_k, cached_v) in enumerate(acquired):
        partial = partials[source_index]
        _row_int8_v_amax_kernel[grid](
            cached_v.qdata, cached_v.scale, block_indices, partial,
            SOURCE_TOKENS=slot_tokens, HEADS=heads,
            HEAD_DIM=head_dim, BLOCK_N=256,
        )
    current_partial = partials[-1]
    _float_v_amax_kernel[grid](
        current, block_indices, current_partial,
        SOURCE_TOKENS=current_tokens, HEADS=heads,
        HEAD_DIM=head_dim, BLOCK_N=256,
    )
    scale = torch.empty(
        (batch, heads, head_dim),
        device=current.device,
        dtype=torch.float32,
    )
    torch.amax(partials, dim=0, out=scale)
    scale.div_(scale_max).clamp_min_(1.0e-8)
    if profiler:
        profiler.profile_end("sparge_v_scale_reduce", scale_marker)

    total_marker = profiler.profile_start(current) if profiler else None
    output = torch.empty(
        (batch, heads, head_dim, padded_total),
        device=current.device, dtype=torch.float8_e4m3fn,
    )
    if padded_total > total_tokens:
        output[..., total_tokens:].zero_()
    marker = profiler.profile_start(current) if profiler else None
    offset = 0
    grid_q = (triton.cdiv(slot_tokens, 16), heads, batch)
    for _cached_k, cached_v in acquired:
        _row_int8_v_fp8_kernel[grid_q](
            cached_v.qdata, cached_v.scale, block_indices,
            scale, output,
            SOURCE_TOKENS=slot_tokens, PADDED_TOTAL=padded_total,
            HEADS=heads, HEAD_DIM=head_dim, OUTPUT_OFFSET=offset,
            SCALE_MAX=scale_max, BLOCK_N=16,
        )
        offset += slot_tokens
    if profiler:
        profiler.profile_end("sparge_v_int8_to_fp8", marker)
    marker = profiler.profile_start(current) if profiler else None
    grid_q = (triton.cdiv(current_tokens, 16), heads, batch)
    _float_v_fp8_kernel[grid_q](
        current, block_indices, scale, output,
        SOURCE_TOKENS=current_tokens, PADDED_TOTAL=padded_total,
        HEADS=heads, HEAD_DIM=head_dim, OUTPUT_OFFSET=offset,
        SCALE_MAX=scale_max, BLOCK_N=16,
    )
    if profiler:
        profiler.profile_end("sparge_v_current_to_fp8", marker)
        profiler.profile_end("sparge_v_native_total", total_marker)
    return output, scale
