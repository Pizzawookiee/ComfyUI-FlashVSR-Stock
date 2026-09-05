"""Native Comfy-Kitchen INT8 sparse attention for FlashVSR.

This backend consumes FlashVSR's existing logical 128x128 LCSA mask and calls
the plain-C INT8 attention ABI used by Zironic/H3-Optimizations' vendored
Comfy-Kitchen-derived sparse kernel.

No H3 Python package is required. Put the compatible DLL in models/flashvsr/.
Faithful INT8 KV caches are repacked in bounded 128-row slabs rather than
materialized as full floating K/V tensors.
The loader prefers h3_int8_attention_v5.dll, then h3_int8_attention.dll, then
a single matching *int8*attention*.dll candidate.

The native library is expected to expose ABI version 4.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from pathlib import Path
import threading

import torch

from .model_paths import MODEL_DIRECTORY
from .qkv import Int8Carrier
from .sparse_backend import SPARSE_BACKEND_OPTION


ABI_VERSION = 4
Q_TILE = 128
KV_TILE = 128
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

_loader_lock = threading.Lock()
_loaded_library = None
_loaded_path = None


def _ptr(tensor):
    return None if tensor is None else tensor.data_ptr()


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _pad_to(length, multiple):
    return ((int(length) + multiple - 1) // multiple) * multiple


def _candidate_library_paths():
    root = Path(MODEL_DIRECTORY)
    preferred = [
        root / "h3_int8_attention_v5.dll",
        root / "h3_int8_attention.dll",
        root / "flashvsr_int8_attention.dll",
        root / "flashvsr_kitchen.dll",
    ]
    existing = [path for path in preferred if path.is_file()]
    if existing:
        return existing

    if root.is_dir():
        matches = sorted(
            path for path in root.glob("*.dll")
            if "int8" in path.name.lower()
            and "attention" in path.name.lower()
        )
        if matches:
            return matches
    return preferred


def _bind_library(library):
    p = ctypes.c_void_p
    i = ctypes.c_int
    i64 = ctypes.c_int64
    f = ctypes.c_float
    sz = ctypes.c_size_t

    library.h3_int8_abi_version.restype = i
    library.h3_int8_abi_version.argtypes = []

    library.h3_int8_last_error.restype = ctypes.c_char_p
    library.h3_int8_last_error.argtypes = []

    library.h3_int8_route_encoding.restype = ctypes.c_char_p
    library.h3_int8_route_encoding.argtypes = []

    library.h3_int8_quantize_qk.restype = i
    library.h3_int8_quantize_qk.argtypes = (
        [p, p, p, p, p, p]
        + [i] * 10
        + [i64] * 6
        + [i, p, sz]
    )

    library.h3_int8_quantize_v.restype = i
    library.h3_int8_quantize_v.argtypes = (
        [p, p, p]
        + [i] * 5
        + [i64] * 3
        + [i, sz]
    )

    # ABI-4 chunked producer/staging entry points. These are what allow the
    # faithful INT8 KV cache to feed Kitchen without reconstructing full K/V.
    library.h3_int8_select_k_anchor.restype = i
    library.h3_int8_select_k_anchor.argtypes = (
        [p, ctypes.POINTER(i), p, p]
        + [i] * 4
        + [i64] * 3
        + [i, sz]
    )
    library.h3_int8_quantize_qk_chunk.restype = i
    library.h3_int8_quantize_qk_chunk.argtypes = (
        [p] * 8 + [i] * 11 + [i64] * 6 + [i, sz]
    )
    library.h3_int8_v_amax_chunk.restype = i
    library.h3_int8_v_amax_chunk.argtypes = (
        [p, p] + [i] * 4 + [i64] * 3 + [i, sz]
    )
    library.h3_int8_quantize_v_chunk_into.restype = i
    library.h3_int8_quantize_v_chunk_into.argtypes = (
        [p, p, p] + [i] * 6 + [i64] * 3 + [i, sz]
    )

    attention_common = [p, p, p, p, p, p, p]
    geometry = [i] * 6
    strides = [i] * 12
    library.h3_int8_sparse_attention.restype = i
    library.h3_int8_sparse_attention.argtypes = (
        attention_common
        + [p, p, i, i, i]
        + geometry
        + strides
        + [i, i, f, i, sz]
    )
    return library


def load_native_library():
    global _loaded_library, _loaded_path
    with _loader_lock:
        if _loaded_library is not None:
            return _loaded_library

        if not torch.cuda.is_available():
            raise RuntimeError(
                "FlashVSR Kitchen Sparse Attention requires NVIDIA CUDA."
            )
        if getattr(torch.version, "hip", None):
            raise RuntimeError(
                "FlashVSR Kitchen Sparse Attention requires NVIDIA CUDA; "
                "ROCm/HIP is not supported by this DLL."
            )

        candidates = _candidate_library_paths()
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            searched = "\n  ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(
                "FlashVSR Kitchen Sparse Attention could not find the native "
                "INT8 attention DLL. Put h3_int8_attention_v5.dll (preferred) "
                "or h3_int8_attention.dll in models/flashvsr.\n"
                f"Looked in:\n  {searched}"
            )

        try:
            library = ctypes.CDLL(str(path))
        except OSError as error:
            raise RuntimeError(
                f"Could not load FlashVSR Kitchen sparse DLL {path}: {error}"
            ) from error

        try:
            library.h3_int8_abi_version.restype = ctypes.c_int
            library.h3_int8_abi_version.argtypes = []
            found_abi = int(library.h3_int8_abi_version())
        except AttributeError as error:
            raise RuntimeError(
                f"{path} is not a compatible H3/Comfy-Kitchen INT8 attention "
                "library: h3_int8_abi_version is missing."
            ) from error

        if found_abi != ABI_VERSION:
            raise RuntimeError(
                f"{path} reports native ABI {found_abi}; this FlashVSR adapter "
                f"expects ABI {ABI_VERSION}."
            )

        try:
            _bind_library(library)
        except AttributeError as error:
            raise RuntimeError(
                f"{path} is missing a required ABI {ABI_VERSION} symbol: {error}"
            ) from error

        _loaded_library = library
        _loaded_path = path
        print(f"[FlashVSR] Kitchen sparse INT8 library loaded: {path}")
        return library


def _check_native(status, what):
    if int(status) == 0:
        return
    library = load_native_library()
    detail = library.h3_int8_last_error()
    detail = detail.decode("utf-8", "replace") if detail else "no detail reported"
    raise RuntimeError(
        f"FlashVSR Kitchen {what} failed (status {int(status)}): {detail}"
    )


def _route_encoding():
    value = load_native_library().h3_int8_route_encoding()
    if not value:
        raise RuntimeError(
            "FlashVSR Kitchen sparse DLL did not report a route encoding."
        )
    encoding = value.decode("ascii", "replace")
    if encoding not in ("absolute", "delta"):
        raise RuntimeError(
            f"Unsupported FlashVSR Kitchen route encoding {encoding!r}."
        )
    return encoding


@dataclass(frozen=True)
class PrequantizedInt8Attention:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    original_head_dim: int
    input_dtype: torch.dtype
    attention_scale: float
    cta_k: int
    anchor_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class BlockSparseRoute:
    indices: torch.Tensor
    counts: torch.Tensor
    q_tile: int = Q_TILE
    kv_tile: int = KV_TILE
    encoding: str = "absolute"

    def _live(self):
        slots = torch.arange(
            self.indices.shape[-1], device=self.indices.device
        )
        return slots < self.counts.to(torch.int64).unsqueeze(-1)

    def to_absolute(self):
        if self.encoding == "absolute":
            return self
        positions = self.indices.to(torch.int64).cumsum(dim=-1)
        indices = torch.where(
            self._live(), positions, torch.zeros_like(positions)
        ).to(torch.int32)
        return replace(
            self, indices=indices.contiguous(), encoding="absolute"
        )

    def to_delta(self):
        if self.encoding == "delta":
            return self
        tiles = self.indices.to(torch.int64)
        previous = torch.cat(
            (torch.zeros_like(tiles[..., :1]), tiles[..., :-1]), dim=-1
        )
        steps = tiles - previous
        indices = torch.where(
            self._live(), steps, torch.zeros_like(steps)
        ).to(torch.int32)
        return replace(self, indices=indices.contiguous(), encoding="delta")

    def for_kernel(self):
        if _route_encoding() == "delta":
            return self.to_delta()
        return self.to_absolute()


def _mask_to_route(mask):
    """Pack FlashVSR's logical [B,H,Qblocks,Kblocks] mask into a native LUT."""
    if mask.ndim != 4:
        raise RuntimeError(
            f"FlashVSR Kitchen expected a rank-4 sparse mask, got {mask.ndim}."
        )
    selected = mask.to(dtype=torch.bool)
    counts = selected.sum(dim=-1, dtype=torch.int32).contiguous()
    if bool((counts == 0).any()):
        raise RuntimeError(
            "FlashVSR Kitchen received an LCSA row with no selected KV blocks."
        )

    kv_tiles = selected.shape[-1]
    candidates = torch.arange(
        kv_tiles, device=selected.device, dtype=torch.int32
    ).view(1, 1, 1, kv_tiles)

    # Selected indices stay in ascending order; rejected entries sort to the
    # tail. The native kernel reads only counts[row] entries.
    sentinel = torch.full(
        (), kv_tiles, device=selected.device, dtype=torch.int32
    )
    packed = torch.where(selected, candidates, sentinel)
    packed = packed.sort(dim=-1).values.contiguous()

    return BlockSparseRoute(
        indices=packed,
        counts=counts,
        q_tile=Q_TILE,
        kv_tile=KV_TILE,
        encoding="absolute",
    )


def _validate_qkv(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError(
            "FlashVSR Kitchen expects HND Q/K/V tensors with rank 4."
        )
    if q.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            f"FlashVSR Kitchen does not support Q dtype {q.dtype}."
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError(
            "FlashVSR Kitchen requires Q, K and V to use the same dtype."
        )
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("FlashVSR Kitchen requires CUDA Q/K/V tensors.")
    if q.shape[-1] != 128:
        raise RuntimeError(
            "FlashVSR Kitchen 128x128 LCSA currently requires head_dim 128; "
            f"received {q.shape[-1]}."
        )


def _prequantize(q, k, v):
    library = load_native_library()
    _validate_qkv(q, k, v)

    # H3/Kitchen's Q carrier is always Q128. FlashVSR's logical LCSA K tile is
    # also 128, so force CTA_K=128 and preserve the mask geometry exactly.
    cta_k = KV_TILE
    original_head_dim = int(q.shape[-1])
    kernel_head_dim = 64 if original_head_dim <= 64 else 128
    input_dtype = q.dtype

    if kernel_head_dim != original_head_dim:
        pad = (0, kernel_head_dim - original_head_dim)
        q = torch.nn.functional.pad(q, pad)
        k = torch.nn.functional.pad(k, pad)
        v = torch.nn.functional.pad(v, pad)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch, q_heads, q_length, _ = q.shape
    _, kv_heads, kv_length, _ = k.shape
    padded_k_length = _pad_to(kv_length, cta_k)

    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = torch.empty_like(k, dtype=torch.int8)
    q_scale = torch.empty(
        batch,
        q_heads,
        ((q_length + Q_TILE - 1) // Q_TILE) * 32,
        dtype=torch.float32,
        device=q.device,
    )
    k_scale = torch.empty(
        batch,
        kv_heads,
        ((kv_length + cta_k - 1) // cta_k) * 4,
        dtype=torch.float32,
        device=q.device,
    )
    v_int8 = torch.empty(
        batch * kv_heads * kernel_head_dim,
        padded_k_length,
        dtype=torch.int8,
        device=q.device,
    )
    v_scale = torch.empty(
        batch * kv_heads * kernel_head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    anchor_indices = torch.empty(
        batch, kv_heads, dtype=torch.int32, device=q.device
    )

    dtype_code = DTYPE_TO_CODE[input_dtype]
    warp_q = 32

    _check_native(
        library.h3_int8_quantize_qk(
            _ptr(q), _ptr(q_int8), _ptr(q_scale),
            _ptr(k), _ptr(k_int8), _ptr(k_scale),
            batch, q_heads, q_length, kv_heads, kv_length, kernel_head_dim,
            Q_TILE, warp_q, cta_k, cta_k,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            dtype_code, _ptr(anchor_indices), _stream(),
        ),
        "Q/K quantization",
    )
    _check_native(
        library.h3_int8_quantize_v(
            _ptr(v), _ptr(v_int8), _ptr(v_scale),
            batch, kv_heads, kv_length, kernel_head_dim, padded_k_length,
            v.stride(0), v.stride(1), v.stride(2),
            dtype_code, _stream(),
        ),
        "V quantization",
    )

    return PrequantizedInt8Attention(
        q=q_int8,
        k=k_int8,
        v=v_int8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        original_head_dim=original_head_dim,
        input_dtype=input_dtype,
        attention_scale=original_head_dim ** -0.5,
        cta_k=cta_k,
        anchor_indices=anchor_indices,
    )


def _attention_geometry(quantized):
    batch, q_heads, q_length, kernel_head_dim = quantized.q.shape
    kv_heads, kv_length = quantized.k.shape[1], quantized.k.shape[2]
    padded_k_length = _pad_to(kv_length, quantized.cta_k)

    output_dtype = (
        torch.bfloat16
        if quantized.input_dtype == torch.float32
        else quantized.input_dtype
    )
    output = torch.empty(
        batch,
        q_heads,
        q_length,
        kernel_head_dim,
        dtype=output_dtype,
        device=quantized.q.device,
    )

    strides = (
        q_heads * q_length * kernel_head_dim,
        kernel_head_dim,
        q_length * kernel_head_dim,
        kv_heads * kv_length * kernel_head_dim,
        kernel_head_dim,
        kv_length * kernel_head_dim,
        kv_heads * kernel_head_dim * padded_k_length,
        kernel_head_dim * padded_k_length,
        padded_k_length,
        q_heads * q_length * kernel_head_dim,
        kernel_head_dim,
        q_length * kernel_head_dim,
    )
    geometry = (
        batch, q_length, kv_length, q_heads, kv_heads, kernel_head_dim
    )
    return output, output_dtype, geometry, strides


def _run_sparse_kernel(q_hnd, k_hnd, v_hnd, mask, profiler=None):
    quant_marker = profiler.profile_start(q_hnd) if profiler else None
    quantized = _prequantize(q_hnd, k_hnd, v_hnd)
    if profiler:
        profiler.profile_end("kitchen_qkv_quant", quant_marker)

    route_marker = profiler.profile_start(mask) if profiler else None
    route = _mask_to_route(mask).for_kernel()
    if profiler:
        profiler.profile_end("kitchen_lut", route_marker)

    if route.kv_tile != quantized.cta_k:
        raise RuntimeError(
            "FlashVSR Kitchen route/carrier KV tile mismatch: "
            f"{route.kv_tile} != {quantized.cta_k}."
        )

    output, output_dtype, geometry, strides = _attention_geometry(quantized)
    slots = int(route.indices.shape[-1])
    library = load_native_library()

    kernel_marker = profiler.profile_start(q_hnd) if profiler else None
    _check_native(
        library.h3_int8_sparse_attention(
            _ptr(quantized.q), _ptr(quantized.k), _ptr(quantized.v),
            _ptr(output),
            _ptr(quantized.q_scale), _ptr(quantized.k_scale),
            _ptr(quantized.v_scale),
            _ptr(route.indices), _ptr(route.counts),
            slots, route.q_tile, quantized.cta_k,
            *geometry,
            *strides,
            quantized.q_scale.stride(0),
            quantized.q_scale.stride(1),
            quantized.attention_scale,
            DTYPE_TO_CODE[output_dtype],
            _stream(),
        ),
        "sparse attention",
    )
    if profiler:
        profiler.profile_end("kitchen_sparse_attention", kernel_marker)

    output = output[..., :quantized.original_head_dim]
    if quantized.input_dtype == torch.float32:
        output = output.float()
    return output


def _compact_rows_hnd(source, token_map, heads, head_dim, dtype):
    """Gather one bounded source slab directly into HND floating rows.

    Int8Carrier is dequantized only for the requested rows. Float/hybrid
    sources are gathered directly. The returned slab is the only floating
    cache staging allocation used by the Kitchen compact path.
    """
    if isinstance(source, Int8Carrier):
        tokens = int(source.shape[1])
        batch = int(source.shape[0])
        qdata = (
            source.qdata.view(batch, tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, token_map)
        )
        scale = (
            source.scale.view(batch, tokens, heads, 1)
            .permute(0, 2, 1, 3)
            .index_select(2, token_map)
        )
        return qdata.to(dtype=dtype).mul_(scale.to(dtype=dtype)).contiguous()

    if not torch.is_tensor(source) or source.ndim != 3:
        raise RuntimeError(
            "FlashVSR Kitchen compact cache contained an unsupported value."
        )
    batch, tokens, channels = source.shape
    if channels != heads * head_dim:
        raise RuntimeError("FlashVSR Kitchen compact cache channel mismatch.")
    return (
        source.view(batch, tokens, heads, head_dim)
        .permute(0, 2, 1, 3)
        .index_select(2, token_map)
        .to(dtype=dtype)
        .contiguous()
    )


def _compact_sources(descriptor, acquired):
    """Chronological K/V sources: compact history followed by current float."""
    for cached_k, cached_v in acquired:
        yield cached_k, cached_v
    yield descriptor.current_k, descriptor.current_v


def _anchor_positions(total_tokens):
    return tuple(sample * (total_tokens - 1) // 8 for sample in range(9))


def _sample_compact_k(
    descriptor, acquired, token_map, heads, head_dim, dtype, device
):
    """Gather Kitchen's nine global K-anchor sample rows without full K."""
    slot_tokens = descriptor.tokens_per_frame * 2
    current_tokens = int(descriptor.current_k.shape[1])
    lengths = [slot_tokens] * len(acquired) + [current_tokens]
    positions = _anchor_positions(sum(lengths))
    samples = []

    # token_map maps each source's block-ordered row to its source row. Each
    # historical slot and the current segment use the same spatial ordering.
    source_pairs = list(_compact_sources(descriptor, acquired))
    for absolute in positions:
        base = 0
        for source_index, length in enumerate(lengths):
            if absolute < base + length:
                local = absolute - base
                source_k = source_pairs[source_index][0]
                mapped = token_map[local:local + 1]
                samples.append(
                    _compact_rows_hnd(
                        source_k, mapped, heads, head_dim, dtype
                    )
                )
                break
            base += length
        else:
            raise RuntimeError("Kitchen K-anchor sample fell outside cache.")
    return torch.cat(samples, dim=2).contiguous(), positions


def _select_k_anchor(
    descriptor, acquired, token_map, heads, head_dim, dtype, device,
    total_tokens,
):
    library = load_native_library()
    samples, positions = _sample_compact_k(
        descriptor, acquired, token_map, heads, head_dim, dtype, device
    )
    sample_positions = torch.tensor(
        positions, dtype=torch.int32, device=device
    )
    values = torch.empty(
        samples.shape[0], heads, head_dim, dtype=dtype, device=device
    )
    indices = torch.empty(
        samples.shape[0], heads, dtype=torch.int32, device=device
    )
    _check_native(
        library.h3_int8_select_k_anchor(
            _ptr(samples),
            ctypes.cast(
                _ptr(sample_positions), ctypes.POINTER(ctypes.c_int)
            ),
            _ptr(values),
            _ptr(indices),
            samples.shape[0],
            heads,
            total_tokens,
            head_dim,
            samples.stride(0),
            samples.stride(1),
            samples.stride(2),
            DTYPE_TO_CODE[dtype],
            _stream(),
        ),
        "K anchor selection",
    )
    return values, indices


def _allocate_compact_carrier(
    batch, heads, q_tokens, k_tokens, head_dim, dtype, device
):
    padded_k = _pad_to(k_tokens, KV_TILE)
    return PrequantizedInt8Attention(
        q=torch.empty(
            batch, heads, q_tokens, head_dim,
            dtype=torch.int8, device=device
        ),
        k=torch.empty(
            batch, heads, k_tokens, head_dim,
            dtype=torch.int8, device=device
        ),
        v=torch.empty(
            batch * heads * head_dim, padded_k,
            dtype=torch.int8, device=device
        ),
        q_scale=torch.empty(
            batch, heads, ((q_tokens + Q_TILE - 1) // Q_TILE) * 32,
            dtype=torch.float32, device=device
        ),
        k_scale=torch.empty(
            batch, heads, ((k_tokens + KV_TILE - 1) // KV_TILE) * 4,
            dtype=torch.float32, device=device
        ),
        v_scale=torch.empty(
            batch * heads * head_dim,
            dtype=torch.float32, device=device
        ),
        original_head_dim=head_dim,
        input_dtype=dtype,
        attention_scale=head_dim ** -0.5,
        cta_k=KV_TILE,
        anchor_indices=None,
    )


def _quantize_qk_chunk_into(
    carrier, q_chunk, k_chunk, anchor_values, anchor_indices,
    q_start, k_start, full_q, full_k,
):
    library = load_native_library()
    _check_native(
        library.h3_int8_quantize_qk_chunk(
            _ptr(q_chunk),
            _ptr(k_chunk),
            _ptr(carrier.q),
            _ptr(carrier.q_scale),
            _ptr(carrier.k),
            _ptr(carrier.k_scale),
            _ptr(anchor_values),
            _ptr(anchor_indices),
            q_chunk.shape[0],
            q_chunk.shape[1],
            q_chunk.shape[2],
            full_q,
            q_start,
            k_chunk.shape[1],
            k_chunk.shape[2],
            full_k,
            k_start,
            q_chunk.shape[-1],
            KV_TILE,
            q_chunk.stride(0),
            q_chunk.stride(1),
            q_chunk.stride(2),
            k_chunk.stride(0),
            k_chunk.stride(1),
            k_chunk.stride(2),
            DTYPE_TO_CODE[q_chunk.dtype],
            _stream(),
        ),
        "chunked Q/K packing",
    )


def _v_amax_update(amax, chunk):
    library = load_native_library()
    _check_native(
        library.h3_int8_v_amax_chunk(
            _ptr(chunk), _ptr(amax),
            chunk.shape[0], chunk.shape[1], chunk.shape[2], chunk.shape[3],
            chunk.stride(0), chunk.stride(1), chunk.stride(2),
            DTYPE_TO_CODE[chunk.dtype], _stream(),
        ),
        "chunked V amax",
    )


def _v_quantize_chunk(carrier, chunk, row_start, padded_k):
    library = load_native_library()
    _check_native(
        library.h3_int8_quantize_v_chunk_into(
            _ptr(chunk), _ptr(carrier.v), _ptr(carrier.v_scale),
            chunk.shape[0], chunk.shape[1], chunk.shape[2],
            int(row_start), chunk.shape[3], padded_k,
            chunk.stride(0), chunk.stride(1), chunk.stride(2),
            DTYPE_TO_CODE[chunk.dtype], _stream(),
        ),
        "chunked V packing",
    )


def _direct_compact_prequantize(
    q_hnd,
    descriptor,
    acquired,
    token_map,
    profiler=None,
):
    """Build Kitchen carriers without a full floating history K/V tensor."""
    batch, heads, q_tokens, head_dim = q_hnd.shape
    dtype = q_hnd.dtype
    device = q_hnd.device
    slot_tokens = descriptor.tokens_per_frame * 2
    current_tokens = int(descriptor.current_k.shape[1])
    total_k = slot_tokens * len(acquired) + current_tokens
    if total_k % KV_TILE:
        raise RuntimeError(
            "FlashVSR Kitchen compact K length must be 128-token aligned."
        )
    if q_tokens % Q_TILE:
        raise RuntimeError(
            "FlashVSR Kitchen compact Q length must be 128-token aligned."
        )

    required = (
        "h3_int8_select_k_anchor",
        "h3_int8_quantize_qk_chunk",
        "h3_int8_v_amax_chunk",
        "h3_int8_quantize_v_chunk_into",
    )
    library = load_native_library()
    missing = [name for name in required if not hasattr(library, name)]
    if missing:
        raise RuntimeError(
            "FlashVSR Kitchen compact path requires the current ABI-4 "
            "chunk producer symbols; missing: " + ", ".join(missing)
        )

    carrier = _allocate_compact_carrier(
        batch, heads, q_tokens, total_k, head_dim, dtype, device
    )

    anchor_marker = profiler.profile_start(q_hnd) if profiler else None
    anchor_values, anchor_indices = _select_k_anchor(
        descriptor, acquired, token_map, heads, head_dim, dtype, device,
        total_k,
    )
    carrier = replace(carrier, anchor_indices=anchor_indices)
    if profiler:
        profiler.profile_end("kitchen_compact_anchor", anchor_marker)

    # Pack Q in 128-row chunks. The ABI packs Q and K together, so pair the Q
    # chunks with the first K source chunks. K will subsequently be overwritten
    # with the same deterministic carrier as each chronological source is
    # packed. This keeps temporary floating storage bounded to two 128-row slabs.
    q_marker = profiler.profile_start(q_hnd) if profiler else None
    first_k_source = acquired[0][0] if acquired else descriptor.current_k
    for q_start in range(0, q_tokens, Q_TILE):
        q_chunk = q_hnd[:, :, q_start:q_start + Q_TILE].contiguous()
        local_start = q_start % (
            slot_tokens if acquired else current_tokens
        )
        local_end = min(
            local_start + KV_TILE,
            slot_tokens if acquired else current_tokens,
        )
        if local_end - local_start != KV_TILE:
            local_start = 0
            local_end = KV_TILE
        mapped = token_map[local_start:local_end]
        k_chunk = _compact_rows_hnd(
            first_k_source, mapped, heads, head_dim, dtype
        )
        _quantize_qk_chunk_into(
            carrier, q_chunk, k_chunk, anchor_values, anchor_indices,
            q_start, 0, q_tokens, total_k,
        )
        del q_chunk, k_chunk
    if profiler:
        profiler.profile_end("kitchen_compact_q_pack", q_marker)

    # Chronological K pack. Every temporary slab is at most 128 rows.
    k_marker = profiler.profile_start(q_hnd) if profiler else None
    k_offset = 0
    for source_k, _source_v in _compact_sources(descriptor, acquired):
        source_tokens = (
            slot_tokens if k_offset < slot_tokens * len(acquired)
            else current_tokens
        )
        for start in range(0, source_tokens, KV_TILE):
            end = min(start + KV_TILE, source_tokens)
            mapped = token_map[start:end]
            k_chunk = _compact_rows_hnd(
                source_k, mapped, heads, head_dim, dtype
            )
            # Reuse a harmless Q128 slab to drive the shared producer ABI.
            q_chunk = q_hnd[:, :, :Q_TILE].contiguous()
            _quantize_qk_chunk_into(
                carrier, q_chunk, k_chunk,
                anchor_values, anchor_indices,
                0, k_offset + start, q_tokens, total_k,
            )
            del q_chunk, k_chunk
        k_offset += source_tokens
    if profiler:
        profiler.profile_end("kitchen_compact_k_pack", k_marker)

    # Kitchen V uses one scale per [B,H,D] across the entire K sequence.
    # First pass accumulates amax from bounded slabs, second pass quantizes
    # those same slabs directly into the final transposed/permuted carrier.
    v_marker = profiler.profile_start(q_hnd) if profiler else None
    amax = torch.zeros(
        batch, heads, head_dim, dtype=torch.float32, device=device
    )
    source_specs = list(_compact_sources(descriptor, acquired))
    for source_index, (_source_k, source_v) in enumerate(source_specs):
        source_tokens = (
            slot_tokens if source_index < len(acquired) else current_tokens
        )
        for start in range(0, source_tokens, KV_TILE):
            end = min(start + KV_TILE, source_tokens)
            v_chunk = _compact_rows_hnd(
                source_v, token_map[start:end], heads, head_dim, dtype
            )
            _v_amax_update(amax, v_chunk)
            del v_chunk

    carrier.v_scale.copy_(
        torch.clamp(amax * (1.0 / 127.0), min=1e-12)
        .reshape(-1)
    )
    del amax
    padded_k = _pad_to(total_k, KV_TILE)
    if padded_k > total_k:
        carrier.v[..., total_k:].zero_()

    v_offset = 0
    for source_index, (_source_k, source_v) in enumerate(source_specs):
        source_tokens = (
            slot_tokens if source_index < len(acquired) else current_tokens
        )
        for start in range(0, source_tokens, KV_TILE):
            end = min(start + KV_TILE, source_tokens)
            v_chunk = _compact_rows_hnd(
                source_v, token_map[start:end], heads, head_dim, dtype
            )
            _v_quantize_chunk(
                carrier, v_chunk, v_offset + start, padded_k
            )
            del v_chunk
        v_offset += source_tokens
    if profiler:
        profiler.profile_end("kitchen_compact_v_pack", v_marker)

    return carrier


def _live_rows_hnd(source, token_map, heads, head_dim):
    """Gather one bounded stock-Wan BNC slab directly into HND layout."""
    if not torch.is_tensor(source) or source.ndim != 3:
        raise RuntimeError("FlashVSR Kitchen live Q/K/V must be rank-3 BNC tensors.")
    batch, tokens, channels = source.shape
    if channels != heads * head_dim:
        raise RuntimeError(
            "FlashVSR Kitchen live Q/K/V channel count does not match heads."
        )
    return (
        source.view(batch, tokens, heads, head_dim)
        .permute(0, 2, 1, 3)
        .index_select(2, token_map)
        .contiguous()
    )


def _select_live_k_anchor(k, k_block_to_original, heads, head_dim):
    """Select Kitchen's global K anchor from nine block-ordered sample rows."""
    library = load_native_library()
    k_tokens = int(k.shape[1])
    positions = _anchor_positions(k_tokens)
    block_positions = torch.tensor(
        positions, dtype=torch.long, device=k.device
    )
    source_positions = k_block_to_original.index_select(0, block_positions)
    samples = _live_rows_hnd(
        k, source_positions, heads, head_dim
    ).contiguous()
    sample_positions = torch.tensor(
        positions, dtype=torch.int32, device=k.device
    )
    values = torch.empty(
        k.shape[0], heads, head_dim, dtype=k.dtype, device=k.device
    )
    indices = torch.empty(
        k.shape[0], heads, dtype=torch.int32, device=k.device
    )
    _check_native(
        library.h3_int8_select_k_anchor(
            _ptr(samples),
            ctypes.cast(
                _ptr(sample_positions), ctypes.POINTER(ctypes.c_int)
            ),
            _ptr(values),
            _ptr(indices),
            k.shape[0],
            heads,
            k_tokens,
            head_dim,
            samples.stride(0),
            samples.stride(1),
            samples.stride(2),
            DTYPE_TO_CODE[k.dtype],
            _stream(),
        ),
        "live K anchor selection",
    )
    return values, indices


def _direct_live_prequantize(
    q,
    k,
    v,
    heads,
    q_block_to_original,
    k_block_to_original,
    profiler=None,
):
    """Pack live/prefill Q/K/V without materializing full HND float tensors."""
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError(
            "FlashVSR Kitchen live Q/K/V must use the same dtype."
        )
    if q.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            f"FlashVSR Kitchen does not support live dtype {q.dtype}."
        )
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("FlashVSR Kitchen live Q/K/V must be CUDA tensors.")

    batch, q_tokens, channels = q.shape
    k_tokens = int(k.shape[1])
    if k.shape != v.shape or k.shape[0] != batch or k.shape[2] != channels:
        raise RuntimeError(
            "FlashVSR Kitchen received incompatible live Q/K/V shapes."
        )
    if channels % heads:
        raise RuntimeError(
            "FlashVSR Kitchen live channel count is not divisible by heads."
        )
    head_dim = channels // heads
    if head_dim != 128:
        raise RuntimeError(
            "FlashVSR Kitchen 128x128 LCSA currently requires head_dim 128; "
            f"received {head_dim}."
        )
    if q_tokens % Q_TILE or k_tokens % KV_TILE:
        raise RuntimeError(
            "FlashVSR Kitchen live Q/K token counts must be 128 aligned."
        )

    required = (
        "h3_int8_select_k_anchor",
        "h3_int8_quantize_qk_chunk",
        "h3_int8_v_amax_chunk",
        "h3_int8_quantize_v_chunk_into",
    )
    library = load_native_library()
    missing = [name for name in required if not hasattr(library, name)]
    if missing:
        raise RuntimeError(
            "FlashVSR Kitchen live producer requires the current ABI-4 "
            "chunk symbols; missing: " + ", ".join(missing)
        )

    carrier = _allocate_compact_carrier(
        batch, heads, q_tokens, k_tokens, head_dim, q.dtype, q.device
    )

    anchor_marker = profiler.profile_start(k) if profiler else None
    anchor_values, anchor_indices = _select_live_k_anchor(
        k, k_block_to_original, heads, head_dim
    )
    carrier = replace(carrier, anchor_indices=anchor_indices)
    if profiler:
        profiler.profile_end("kitchen_live_anchor", anchor_marker)

    # Q pack: one 128-row slab at a time. Pair each Q slab with the first K
    # slab because the ABI packs Q and K together. K[0:128] is deterministically
    # overwritten; the full K carrier is populated in the following loop.
    pack_marker = profiler.profile_start(q) if profiler else None
    first_k_map = k_block_to_original[:KV_TILE]
    first_k_chunk = _live_rows_hnd(
        k, first_k_map, heads, head_dim
    )
    for q_start in range(0, q_tokens, Q_TILE):
        q_map = q_block_to_original[q_start:q_start + Q_TILE]
        q_chunk = _live_rows_hnd(
            q, q_map, heads, head_dim
        )
        _quantize_qk_chunk_into(
            carrier,
            q_chunk,
            first_k_chunk,
            anchor_values,
            anchor_indices,
            q_start,
            0,
            q_tokens,
            k_tokens,
        )
        del q_chunk
    del first_k_chunk

    # K pack: populate each 128-row block directly into the final carrier.
    q0_map = q_block_to_original[:Q_TILE]
    q0_chunk = _live_rows_hnd(q, q0_map, heads, head_dim)
    for k_start in range(0, k_tokens, KV_TILE):
        k_map = k_block_to_original[k_start:k_start + KV_TILE]
        k_chunk = _live_rows_hnd(
            k, k_map, heads, head_dim
        )
        _quantize_qk_chunk_into(
            carrier,
            q0_chunk,
            k_chunk,
            anchor_values,
            anchor_indices,
            0,
            k_start,
            q_tokens,
            k_tokens,
        )
        del k_chunk
    del q0_chunk
    if profiler:
        profiler.profile_end("kitchen_live_qk_pack", pack_marker)

    # V pack: first pass determines the global [B,H,D] scale. Second pass
    # writes each 128-row slab straight into Kitchen's final INT8 carrier.
    v_marker = profiler.profile_start(v) if profiler else None
    amax = torch.zeros(
        batch, heads, head_dim, dtype=torch.float32, device=v.device
    )
    for start in range(0, k_tokens, KV_TILE):
        v_map = k_block_to_original[start:start + KV_TILE]
        v_chunk = _live_rows_hnd(
            v, v_map, heads, head_dim
        )
        _v_amax_update(amax, v_chunk)
        del v_chunk

    carrier.v_scale.copy_(
        torch.clamp(amax * (1.0 / 127.0), min=1e-12)
        .reshape(-1)
    )
    del amax

    padded_k = _pad_to(k_tokens, KV_TILE)
    if padded_k > k_tokens:
        carrier.v[..., k_tokens:].zero_()

    for start in range(0, k_tokens, KV_TILE):
        v_map = k_block_to_original[start:start + KV_TILE]
        v_chunk = _live_rows_hnd(
            v, v_map, heads, head_dim
        )
        _v_quantize_chunk(
            carrier, v_chunk, start, padded_k
        )
        del v_chunk
    if profiler:
        profiler.profile_end("kitchen_live_v_pack", v_marker)

    return carrier


class FlashVSRKitchenBackend:
    """Execute FlashVSR's prescribed LCSA mask with native Kitchen INT8."""

    flashvsr_block_sparse = True

    def __init__(self):
        # Fail at node execution rather than many seconds into sampling.
        load_native_library()

    @staticmethod
    def _restore(output_hnd, q, heads, q_block_to_original, profiler=None):
        batch, q_tokens, channels = q.shape
        head_dim = channels // heads
        marker = profiler.profile_start(q) if profiler else None
        attended = torch.empty_like(q)
        output_nhd = output_hnd.permute(0, 2, 1, 3)
        if output_nhd.dtype != q.dtype:
            output_nhd = output_nhd.to(dtype=q.dtype)
        attended.view(batch, q_tokens, heads, head_dim).index_copy_(
            1, q_block_to_original, output_nhd
        )
        if profiler:
            profiler.profile_end("kitchen_restore", marker)
        return attended

    @staticmethod
    def _validate_mask(mask, batch, heads, q_tokens, k_tokens):
        if q_tokens % Q_TILE or k_tokens % KV_TILE:
            raise RuntimeError(
                "FlashVSR Kitchen requires Q/K token counts aligned to "
                "128-token logical LCSA blocks."
            )
        expected = (
            batch,
            heads,
            q_tokens // Q_TILE,
            k_tokens // KV_TILE,
        )
        if tuple(mask.shape) != expected:
            raise RuntimeError(
                "FlashVSR Kitchen received an unexpected LCSA mask shape: "
                f"{tuple(mask.shape)} != {expected}."
            )

    def run_flashvsr(
        self,
        q,
        k,
        v,
        heads,
        mask,
        q_block_to_original,
        k_block_to_original,
        profiler=None,
    ):
        """Run live/prefill LCSA through bounded Kitchen carrier production."""
        if q.device.type != "cuda":
            raise RuntimeError(
                "FlashVSR Kitchen Sparse Attention requires CUDA."
            )
        batch, q_tokens, channels = q.shape
        if channels % heads:
            raise RuntimeError(
                "FlashVSR Kitchen received a channel count not divisible by "
                "the attention head count."
            )
        if (
            k.shape != v.shape
            or k.shape[0] != batch
            or k.shape[2] != channels
        ):
            raise RuntimeError(
                "FlashVSR Kitchen received incompatible Q/K/V shapes."
            )
        k_tokens = int(k.shape[1])
        self._validate_mask(
            mask, batch, heads, q_tokens, k_tokens
        )

        # No complete floating q_hnd/k_hnd/v_hnd copies: gather one logical
        # 128-row block at a time directly into the native Kitchen carriers.
        quantized = _direct_live_prequantize(
            q,
            k,
            v,
            heads,
            q_block_to_original,
            k_block_to_original,
            profiler,
        )

        route_marker = profiler.profile_start(mask) if profiler else None
        route = _mask_to_route(mask).for_kernel()
        if profiler:
            profiler.profile_end("kitchen_lut", route_marker)

        output, output_dtype, geometry, strides = _attention_geometry(
            quantized
        )
        library = load_native_library()
        kernel_marker = profiler.profile_start(q) if profiler else None
        _check_native(
            library.h3_int8_sparse_attention(
                _ptr(quantized.q),
                _ptr(quantized.k),
                _ptr(quantized.v),
                _ptr(output),
                _ptr(quantized.q_scale),
                _ptr(quantized.k_scale),
                _ptr(quantized.v_scale),
                _ptr(route.indices),
                _ptr(route.counts),
                int(route.indices.shape[-1]),
                route.q_tile,
                quantized.cta_k,
                *geometry,
                *strides,
                quantized.q_scale.stride(0),
                quantized.q_scale.stride(1),
                quantized.attention_scale,
                DTYPE_TO_CODE[output_dtype],
                _stream(),
            ),
            "live sparse attention",
        )
        if profiler:
            profiler.profile_end(
                "kitchen_sparse_attention", kernel_marker
            )

        output = output[..., :channels // heads]
        if q.dtype == torch.float32:
            output = output.float()
        attended = self._restore(
            output, q, heads, q_block_to_original, profiler
        )
        del output, quantized
        return attended

    def run_flashvsr_compact(
        self,
        q,
        descriptor,
        cache,
        heads,
        mask,
        q_block_to_original,
        slot_block_to_original,
        profiler=None,
    ):
        """Consume faithful compact cache without materializing full float K/V."""
        if q.device.type != "cuda":
            raise RuntimeError(
                "FlashVSR Kitchen Sparse Attention requires CUDA."
            )

        batch, q_tokens, channels = q.shape
        if channels % heads:
            raise RuntimeError("Invalid FlashVSR Kitchen attention head count.")
        head_dim = channels // heads

        total_tokens = (
            descriptor.history_frames * descriptor.tokens_per_frame
            + descriptor.current_k.shape[1]
        )
        self._validate_mask(
            mask, batch, heads, q_tokens, total_tokens
        )

        layout_marker = profiler.profile_start(q) if profiler else None
        q_hnd = (
            q.view(batch, q_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, q_block_to_original)
            .contiguous()
        )
        if profiler:
            profiler.profile_end("kitchen_layout_qkv", layout_marker)

        transfer_marker = profiler.profile_start(q_hnd) if profiler else None
        with cache.acquire_compact_slots(
            descriptor, q_hnd.device
        ) as acquired:
            if profiler:
                profiler.profile_end(
                    "kitchen_compact_h2d", transfer_marker
                )
            quantized = _direct_compact_prequantize(
                q_hnd,
                descriptor,
                acquired,
                slot_block_to_original,
                profiler,
            )
            route_marker = profiler.profile_start(mask) if profiler else None
            route = _mask_to_route(mask).for_kernel()
            if profiler:
                profiler.profile_end("kitchen_lut", route_marker)

            output, output_dtype, geometry, strides = _attention_geometry(
                quantized
            )
            library = load_native_library()
            kernel_marker = (
                profiler.profile_start(q_hnd) if profiler else None
            )
            _check_native(
                library.h3_int8_sparse_attention(
                    _ptr(quantized.q),
                    _ptr(quantized.k),
                    _ptr(quantized.v),
                    _ptr(output),
                    _ptr(quantized.q_scale),
                    _ptr(quantized.k_scale),
                    _ptr(quantized.v_scale),
                    _ptr(route.indices),
                    _ptr(route.counts),
                    int(route.indices.shape[-1]),
                    route.q_tile,
                    quantized.cta_k,
                    *geometry,
                    *strides,
                    quantized.q_scale.stride(0),
                    quantized.q_scale.stride(1),
                    quantized.attention_scale,
                    DTYPE_TO_CODE[output_dtype],
                    _stream(),
                ),
                "compact sparse attention",
            )
            if profiler:
                profiler.profile_end(
                    "kitchen_sparse_attention", kernel_marker
                )

        output = output[..., :head_dim]
        if q.dtype == torch.float32:
            output = output.float()
        attended = self._restore(
            output, q, heads, q_block_to_original, profiler
        )
        del output, quantized, q_hnd
        return attended


def apply_kitchen_backend(model):
    """Attach Kitchen as FlashVSR's private sparse route."""
    patched = model.clone()
    transformer_options = patched.model_options.setdefault(
        "transformer_options", {}
    )
    backend = FlashVSRKitchenBackend()
    transformer_options[SPARSE_BACKEND_OPTION] = backend

    # Match apply_sparge_backend(): patch order remains immaterial.
    override = transformer_options.get("optimized_attention_override")
    runtime = getattr(override, "flashvsr_runtime", None)
    if runtime is not None:
        runtime.set_sparse_attention_backend(backend)
    return patched
