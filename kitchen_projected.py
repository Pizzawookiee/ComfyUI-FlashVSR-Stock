"""Direct stock-Wan projection into Kitchen INT8 sparse carriers.

FlashVSR/Kitchen-specific low-VRAM self-attention.

Both the six-frame faithful prefill and two-frame faithful continuations avoid
complete floating Q/K/V tensors:
- K anchor selection projects only the current rows among Kitchen's nine global
  samples; historical sample rows are dequantized directly from the compact KV
  cache.
- V is projected exactly once into a compact row-INT8 staging/cache carrier
  while Kitchen's global [B,H,D] amax is accumulated.
- Q/K are projected in bounded block-order slabs directly into final Kitchen
  carriers. K is simultaneously packed into the ordinary faithful cache carrier.
- Historical compact K/V are dequantized only in bounded 128/1024-row slabs
  while being repacked into the combined Kitchen carrier.
- The already-compact current K/V carriers are committed directly to CPU/AIMDO;
  runtime.commit() never sees complete floating current K/V.
"""

from __future__ import annotations

from contextlib import ExitStack
import ctypes
from types import SimpleNamespace

import torch

from comfy.ldm.flux.math import apply_rope1

from . import kitchen_sparse_backend as kb
from .qkv import Int8Carrier


PROJECT_CHUNK = 1024


def _gather_freqs(freqs, indices, sequence):
    if not torch.is_tensor(freqs):
        return freqs
    for axis, size in enumerate(freqs.shape):
        if int(size) == int(sequence):
            return freqs.index_select(axis, indices)
    return freqs


def _project_k(module, input_rows, freq_rows, heads, head_dim):
    batch, rows = input_rows.shape[:2]
    value = module.norm_k(module.k(input_rows))
    value = value.view(batch, rows, heads, head_dim)
    return apply_rope1(value, freq_rows)


def _project_q(module, input_rows, freq_rows, heads, head_dim):
    batch, rows = input_rows.shape[:2]
    value = module.norm_q(module.q(input_rows))
    value = value.view(batch, rows, heads, head_dim)
    return apply_rope1(value, freq_rows)


def _project_v(module, input_rows, heads, head_dim):
    batch, rows = input_rows.shape[:2]
    return module.v(input_rows).view(batch, rows, heads, head_dim)


def _to_hnd(value):
    return value.permute(0, 2, 1, 3).contiguous()


def _allocate_row_carrier(batch, tokens, channels, heads, dtype, device):
    head_dim = channels // heads
    return Int8Carrier(
        torch.empty(
            batch * tokens * heads,
            head_dim,
            dtype=torch.int8,
            device=device,
        ),
        torch.empty(
            batch * tokens * heads,
            1,
            dtype=torch.float32,
            device=device,
        ),
        (batch, tokens, channels),
        dtype,
        head_dim,
    )


def _quantize_rows_into(carrier, values_bnhd, token_indices):
    """Write BF16/FP16 [B,N,H,D] rows into an Int8Carrier in stock token order."""
    batch, rows, heads, head_dim = values_bnhd.shape
    total = int(carrier.shape[1])
    qdata = carrier.qdata.view(batch, total, heads, head_dim)
    scales = carrier.scale.view(batch, total, heads, 1)

    flat = values_bnhd.reshape(-1, head_dim)
    scale = flat.float().abs_().amax(dim=-1, keepdim=True)
    scale.div_(127.0).clamp_min_(1.0e-8)
    quantized = flat.float().div_(scale)
    quantized.round_().clamp_(-127, 127)
    quantized = quantized.to(torch.int8).view(batch, rows, heads, head_dim)
    scale = scale.view(batch, rows, heads, 1)

    qdata.index_copy_(1, token_indices, quantized)
    scales.index_copy_(1, token_indices, scale)
    del quantized, scale


def _slice_carrier(carrier, start, end):
    batch, tokens, channels = carrier.shape
    heads = carrier.qdata.shape[0] // (batch * tokens)
    head_dim = carrier.head_dim
    qdata = (
        carrier.qdata.view(batch, tokens, heads, head_dim)
        [:, start:end]
        .contiguous()
        .view(-1, head_dim)
    )
    scale = (
        carrier.scale.view(batch, tokens, heads, 1)
        [:, start:end]
        .contiguous()
        .view(-1, 1)
    )
    return Int8Carrier(
        qdata, scale, (batch, end - start, channels),
        carrier.dtype, head_dim,
    )


def _run_carrier(carrier, mask):
    route = kb._mask_to_route(mask).for_kernel()
    output, output_dtype, geometry, strides = kb._attention_geometry(carrier)
    library = kb.load_native_library()
    kb._check_native(
        library.h3_int8_sparse_attention(
            kb._ptr(carrier.q),
            kb._ptr(carrier.k),
            kb._ptr(carrier.v),
            kb._ptr(output),
            kb._ptr(carrier.q_scale),
            kb._ptr(carrier.k_scale),
            kb._ptr(carrier.v_scale),
            kb._ptr(route.indices),
            kb._ptr(route.counts),
            int(route.indices.shape[-1]),
            route.q_tile,
            carrier.cta_k,
            *geometry,
            *strides,
            carrier.q_scale.stride(0),
            carrier.q_scale.stride(1),
            carrier.attention_scale,
            kb.DTYPE_TO_CODE[output_dtype],
            kb._stream(),
        ),
        "projected sparse attention",
    )
    return output[..., :carrier.original_head_dim]


def _chronological_slots(cache, block_index):
    slots = cache.entries.get(int(block_index))
    if slots is None:
        raise RuntimeError(
            f"FlashVSR projected Kitchen cache for Wan block {block_index} "
            "was not initialized by the first six-frame model call."
        )
    return tuple(
        slots[(cache.write_slot + relative) % cache.slot_count]
        for relative in range(cache.slot_count)
    )


def _acquire_history(cache, chronological, device, stack):
    acquired = []
    for cached_k, cached_v, _summary in chronological:
        k = stack.enter_context(cache._acquire_compact(cached_k, device))
        v = stack.enter_context(cache._acquire_compact(cached_v, device))
        acquired.append((k, v))
    return tuple(acquired)


def _anchor_positions(total_tokens):
    return tuple(sample * (total_tokens - 1) // 8 for sample in range(9))


def _select_global_anchor(
    module,
    x,
    freqs,
    gather_input,
    acquired,
    slot_block_to_original,
    current_block_to_original,
    slot_tokens,
    current_tokens,
    heads,
    head_dim,
):
    """Build exactly nine K samples; project only samples from current tokens."""
    positions = _anchor_positions(
        len(acquired) * slot_tokens + current_tokens
    )
    samples = []
    device = x.device
    dtype = x.dtype
    history_tokens = len(acquired) * slot_tokens

    for absolute in positions:
        if absolute < history_tokens:
            source_index = absolute // slot_tokens
            local = absolute % slot_tokens
            source_k = acquired[source_index][0]
            mapped = slot_block_to_original[local:local + 1]
            samples.append(
                kb._compact_rows_hnd(
                    source_k, mapped, heads, head_dim, dtype
                )
            )
        else:
            local = absolute - history_tokens
            source_index = current_block_to_original[local:local + 1]
            current = gather_input(source_index)
            freq_rows = _gather_freqs(freqs, source_index, current_tokens)
            projected = _project_k(
                module, current, freq_rows, heads, head_dim
            )
            samples.append(_to_hnd(projected))
            del current, projected

    samples = torch.cat(samples, dim=2).contiguous()
    sample_positions = torch.tensor(
        positions, dtype=torch.int32, device=device
    )
    values = torch.empty(
        samples.shape[0], heads, head_dim, dtype=dtype, device=device
    )
    indices = torch.empty(
        samples.shape[0], heads, dtype=torch.int32, device=device
    )
    library = kb.load_native_library()
    kb._check_native(
        library.h3_int8_select_k_anchor(
            kb._ptr(samples),
            ctypes.cast(
                kb._ptr(sample_positions), ctypes.POINTER(ctypes.c_int)
            ),
            kb._ptr(values),
            kb._ptr(indices),
            samples.shape[0],
            heads,
            history_tokens + current_tokens,
            head_dim,
            samples.stride(0),
            samples.stride(1),
            samples.stride(2),
            kb.DTYPE_TO_CODE[dtype],
            kb._stream(),
        ),
        "projected global K anchor selection",
    )
    del samples
    return values, indices


def _stage_current_v_once(
    module, x, gather_input, heads, head_dim
):
    """One V GEMM pass: compact cache staging + Kitchen global amax."""
    batch, tokens, channels = x.shape
    carrier = _allocate_row_carrier(
        batch, tokens, channels, heads, x.dtype, x.device
    )
    amax = torch.zeros(
        batch, heads, head_dim, dtype=torch.float32, device=x.device
    )
    for start in range(0, tokens, PROJECT_CHUNK):
        end = min(start + PROJECT_CHUNK, tokens)
        source_indices = torch.arange(
            start, end, device=x.device, dtype=torch.long
        )
        current = gather_input(source_indices)
        values = _project_v(module, current, heads, head_dim)
        _quantize_rows_into(carrier, values, source_indices)
        kb._v_amax_update(amax, _to_hnd(values))
        del current, values
    return carrier, amax


def _pack_history_k(
    kitchen, acquired, slot_map, slot_tokens, total_k,
    q_dummy, anchor_values, anchor_indices, heads, head_dim, dtype
):
    k_offset = 0
    for source_k, _source_v in acquired:
        for start in range(0, slot_tokens, kb.KV_TILE):
            end = min(start + kb.KV_TILE, slot_tokens)
            k_chunk = kb._compact_rows_hnd(
                source_k, slot_map[start:end], heads, head_dim, dtype
            )
            kb._quantize_qk_chunk_into(
                kitchen,
                q_dummy,
                k_chunk,
                anchor_values,
                anchor_indices,
                0,
                k_offset + start,
                kitchen.q.shape[2],
                total_k,
            )
            del k_chunk
        k_offset += slot_tokens


def _pack_history_v(
    kitchen, acquired, slot_map, slot_tokens, heads, head_dim, dtype
):
    amax = torch.zeros(
        kitchen.q.shape[0], heads, head_dim,
        dtype=torch.float32, device=kitchen.q.device
    )
    for _source_k, source_v in acquired:
        for start in range(0, slot_tokens, kb.KV_TILE):
            end = min(start + kb.KV_TILE, slot_tokens)
            chunk = kb._compact_rows_hnd(
                source_v, slot_map[start:end], heads, head_dim, dtype
            )
            kb._v_amax_update(amax, chunk)
            del chunk
    return amax


def _write_combined_v(
    kitchen,
    acquired,
    current_v,
    slot_map,
    current_map,
    slot_tokens,
    heads,
    head_dim,
    dtype,
):
    total_k = int(kitchen.k.shape[2])
    padded_k = kb._pad_to(total_k, kb.KV_TILE)
    offset = 0
    for _source_k, source_v in acquired:
        for start in range(0, slot_tokens, kb.KV_TILE):
            end = min(start + kb.KV_TILE, slot_tokens)
            chunk = kb._compact_rows_hnd(
                source_v, slot_map[start:end], heads, head_dim, dtype
            )
            kb._v_quantize_chunk(kitchen, chunk, offset + start, padded_k)
            del chunk
        offset += slot_tokens

    current_tokens = int(current_v.shape[1])
    for start in range(0, current_tokens, kb.KV_TILE):
        end = min(start + kb.KV_TILE, current_tokens)
        chunk = kb._compact_rows_hnd(
            current_v,
            current_map[start:end],
            heads,
            head_dim,
            dtype,
        )
        kb._v_quantize_chunk(kitchen, chunk, offset + start, padded_k)
        del chunk


class _TensorMeta:
    """Enough tensor metadata for FlashVSRKVCache._initialize_storage()."""

    def __init__(self, shape, dtype, device):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)

    def element_size(self):
        return torch.empty((), dtype=self.dtype).element_size()


def _cpu_pair(cache, k_carrier, v_carrier):
    if cache.async_writer is not None and cache.cache_format == "int8":
        return cache.async_writer.submit_pair(k_carrier, v_carrier)
    return (
        cache._carrier_to(k_carrier, torch.device("cpu")),
        cache._carrier_to(v_carrier, torch.device("cpu")),
    )


def _wrap_new_pair(cache, cpu_k, cpu_v, gpu_k, gpu_v):
    if cache.aimdo_controller is None:
        return cpu_k, cpu_v
    return (
        cache.aimdo_controller.wrap(cpu_k, gpu_k),
        cache.aimdo_controller.wrap(cpu_v, gpu_v),
    )


def _commit_compact_pair(
    runtime,
    block_index,
    k_carrier,
    v_carrier,
    current_summary,
    k_reference_mean,
):
    """Commit already-compact K/V without runtime.commit() or float tensors."""
    cache = runtime.kv_cache
    tokens_per_frame = (
        (runtime.video.height // 16) * (runtime.video.width // 16)
    )
    slot_tokens = cache.SLOT_FRAMES * tokens_per_frame

    if cache.initial_chunk:
        if cache.total_cache_bytes == 0:
            meta = _TensorMeta(
                (k_carrier.shape[0], slot_tokens, k_carrier.shape[2]),
                k_carrier.dtype,
                k_carrier.device,
            )
            cache._initialize_storage(meta, slot_tokens, runtime._projected_heads)

        history_tokens = cache.history_frames * tokens_per_frame
        history_start = k_carrier.shape[1] - history_tokens
        slots = []
        for slot_index in range(cache.slot_count):
            start = history_start + slot_index * slot_tokens
            end = start + slot_tokens
            gpu_k = _slice_carrier(k_carrier, start, end)
            gpu_v = _slice_carrier(v_carrier, start, end)
            cpu_k, cpu_v = _cpu_pair(cache, gpu_k, gpu_v)
            cpu_k, cpu_v = _wrap_new_pair(
                cache, cpu_k, cpu_v, gpu_k, gpu_v
            )
            temporal_index = start // slot_tokens
            summary = (
                current_summary[:, temporal_index:temporal_index + 1]
                .to(device="cpu", non_blocking=False)
                .contiguous()
            )
            slots.append((cpu_k, cpu_v, summary))
        cache.k_reference_means[int(block_index)] = (
            k_reference_mean.to(device="cpu", non_blocking=False).contiguous()
        )
        cache.pending_entries[int(block_index)] = slots
    else:
        if k_carrier.shape[1] != slot_tokens:
            raise RuntimeError(
                "Projected Kitchen continuation cache requires exactly "
                "two current latent frames."
            )
        cached_k, cached_v, _old_summary = (
            cache.entries[int(block_index)][cache.write_slot]
        )
        cpu_k, cpu_v = _cpu_pair(cache, k_carrier, v_carrier)
        pending = []
        for cached, cpu_value, gpu_source in (
            (cached_k, cpu_k, k_carrier),
            (cached_v, cpu_v, v_carrier),
        ):
            from .aimdo_cache import is_aimdo_value
            if is_aimdo_value(cached):
                pending.append((
                    cached,
                    cached.prepare_update(cpu_value, gpu_source),
                ))
            else:
                pending.append(cpu_value)
        pending.append(
            current_summary.to(device="cpu", non_blocking=False).contiguous()
        )
        cache.pending_entries[int(block_index)] = tuple(pending)

    cache.committed_blocks.add(int(block_index))


def run_projected_kitchen_attention(
    backend,
    module,
    x,
    freqs,
    runtime,
    transformer_options,
    gather_input,
):
    """Direct Kitchen self-attention for prefill and faithful continuations."""
    options = transformer_options or {}
    block_index = int(options.get("block_index", -1))
    if block_index < 0 or block_index in runtime.seen_self_attention_blocks:
        return None

    spatial_tokens = (
        (runtime.video.height // 16) * (runtime.video.width // 16)
    )
    frames = int(runtime.current_latent_frames)
    current_tokens = int(x.shape[1])
    if current_tokens != frames * spatial_tokens or frames % 2:
        return None

    cache = runtime.kv_cache
    cache_active = cache.participates(block_index)
    heads = int(module.num_heads)
    head_dim = int(module.head_dim)
    if head_dim != 128 or x.shape[-1] != heads * head_dim:
        return None
    if current_tokens % kb.Q_TILE:
        return None

    height = runtime.video.height // 16
    width = runtime.video.width // 16
    grid_h = height // 8
    grid_w = width // 8
    current_temporal = frames // 2
    spatial_blocks = grid_h * grid_w
    current_blocks = current_temporal * spatial_blocks
    current_map, _ = runtime._lcsa_token_indices(
        frames, height, width, x.device
    )
    slot_map, _ = runtime._lcsa_token_indices(
        cache.SLOT_FRAMES, height, width, x.device
    )

    chronological = ()
    acquired = ()
    stack = ExitStack()
    try:
        if cache_active and not cache.initial_chunk:
            chronological = _chronological_slots(cache, block_index)
            acquired = _acquire_history(
                cache, chronological, x.device, stack
            )

        slot_tokens = cache.SLOT_FRAMES * spatial_tokens
        history_tokens = len(acquired) * slot_tokens
        total_k = history_tokens + current_tokens
        if total_k % kb.KV_TILE:
            return None

        kitchen = kb._allocate_compact_carrier(
            x.shape[0],
            heads,
            current_tokens,
            total_k,
            head_dim,
            x.dtype,
            x.device,
        )

        # One V projection pass. This compact row carrier later becomes the
        # faithful current-V cache entry and is also the source for Kitchen V.
        current_v, current_v_amax = _stage_current_v_once(
            module, x, gather_input, heads, head_dim
        )

        anchor_values, anchor_indices = _select_global_anchor(
            module,
            x,
            freqs,
            gather_input,
            acquired,
            slot_map,
            current_map,
            slot_tokens,
            current_tokens,
            heads,
            head_dim,
        )
        kitchen = kb.replace(
            kitchen, anchor_indices=anchor_indices
        )

        # Combined V scale must include history plus current, but no V GEMM is
        # repeated. Historical rows are bounded dequantizations from cache.
        history_amax = _pack_history_v(
            kitchen, acquired, slot_map, slot_tokens,
            heads, head_dim, x.dtype
        )
        history_amax.copy_(torch.maximum(history_amax, current_v_amax))
        kitchen.v_scale.copy_(
            torch.clamp(
                history_amax * (1.0 / 127.0), min=1e-12
            ).reshape(-1)
        )
        del history_amax, current_v_amax

        current_k = _allocate_row_carrier(
            x.shape[0], current_tokens, x.shape[2],
            heads, x.dtype, x.device
        )
        q_pool = torch.empty(
            x.shape[0], current_blocks, heads, head_dim,
            dtype=x.dtype, device=x.device
        )
        current_k_pool = torch.empty_like(q_pool)
        k_sum = torch.zeros(
            x.shape[0], heads, head_dim,
            dtype=torch.float32, device=x.device
        )

        # Native Q/K producer packs Q and K together. Historical K is packed
        # with a bounded Q0 dummy after current Q has been produced.
        q0 = None
        for start in range(0, current_tokens, PROJECT_CHUNK):
            end = min(start + PROJECT_CHUNK, current_tokens)
            source_indices = current_map[start:end]
            current = gather_input(source_indices)
            freq_rows = _gather_freqs(
                freqs, source_indices, current_tokens
            )
            q_hnd = _to_hnd(
                _project_q(module, current, freq_rows, heads, head_dim)
            )
            k_value = _project_k(
                module, current, freq_rows, heads, head_dim
            )
            k_hnd = _to_hnd(k_value)

            if q0 is None:
                q0 = q_hnd[:, :, :kb.Q_TILE].contiguous()

            # Current K sits after historical K in the combined Kitchen carrier.
            kb._quantize_qk_chunk_into(
                kitchen,
                q_hnd,
                k_hnd,
                anchor_values,
                anchor_indices,
                start,
                history_tokens + start,
                current_tokens,
                total_k,
            )
            _quantize_rows_into(current_k, k_value, source_indices)

            blocks = (end - start) // kb.Q_TILE
            block_start = start // kb.Q_TILE
            block_end = block_start + blocks
            q_pool[:, block_start:block_end].copy_(
                q_hnd.view(
                    x.shape[0], heads, blocks, kb.Q_TILE, head_dim
                ).mean(3).permute(0, 2, 1, 3)
            )
            current_k_pool[:, block_start:block_end].copy_(
                k_hnd.view(
                    x.shape[0], heads, blocks, kb.KV_TILE, head_dim
                ).mean(3).permute(0, 2, 1, 3)
            )
            k_sum.add_(k_hnd.float().sum(dim=2))
            del current, q_hnd, k_hnd, k_value

        if q0 is None:
            return None
        _pack_history_k(
            kitchen,
            acquired,
            slot_map,
            slot_tokens,
            total_k,
            q0,
            anchor_values,
            anchor_indices,
            heads,
            head_dim,
            x.dtype,
        )
        del q0

        _write_combined_v(
            kitchen,
            acquired,
            current_v,
            slot_map,
            current_map,
            slot_tokens,
            heads,
            head_dim,
            x.dtype,
        )

        # Routing summaries are tiny. Historical summaries are already CPU
        # cache metadata; only the current summary was built from projected K.
        q_pool_view = q_pool.view(
            x.shape[0], current_temporal,
            spatial_blocks, heads, head_dim
        )
        current_k_summary = current_k_pool.view_as(q_pool_view)
        if acquired:
            cached_pools = tuple(
                slot[2].to(
                    device=x.device, dtype=x.dtype, non_blocking=False
                )
                for slot in chronological
            )
            k_pool = torch.cat(
                (*cached_pools, current_k_summary), dim=1
            )
        else:
            k_pool = current_k_summary

        k_temporal = k_pool.shape[1]
        mask = runtime._lcsa_mask_from_pools(
            q_pool_view, k_pool, grid_h, grid_w
        ).reshape(
            x.shape[0],
            heads,
            current_blocks,
            k_temporal * spatial_blocks,
        )
        del q_pool, q_pool_view, k_pool

        output_hnd = _run_carrier(kitchen, mask)
        del mask, anchor_values, kitchen
        output = backend._restore(
            output_hnd, x, heads, current_map, profiler=runtime
        )
        del output_hnd

        if cache_active:
            runtime._projected_heads = heads
            try:
                _commit_compact_pair(
                    runtime,
                    block_index,
                    current_k,
                    current_v,
                    current_k_summary,
                    k_sum.div(float(current_tokens)),
                )
            finally:
                del runtime._projected_heads
        del current_k, current_v, current_k_pool, k_sum

        runtime.seen_self_attention_blocks.add(block_index)
        return module.o(output)
    finally:
        stack.close()
