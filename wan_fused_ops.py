"""Optional Triton row kernels for FlashVSR's streamed Wan blocks.

The public helpers are conservative: they only take over layouts covered by
simple broadcast modulation/gating and permanently fall back to PyTorch if
Triton is unavailable or a kernel fails on the installed stack.
"""

from __future__ import annotations

import torch

_TRITON_DISABLED = False
_TRITON_IMPORT_ERROR = None
try:
    import triton
    import triton.language as tl
except Exception as error:  # Windows installs may intentionally omit Triton.
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = error


if triton is not None:
    @triton.jit
    def _layer_norm_modulate_kernel(
        x_ptr,
        shift_ptr,
        scale_ptr,
        out_ptr,
        rows_per_mod: tl.constexpr,
        dim: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < dim
        base = row * dim
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / dim
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / dim
        inv_std = tl.rsqrt(var + eps)
        mod_row = row // rows_per_mod
        mod_base = mod_row * dim
        shift = tl.load(shift_ptr + mod_base + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + mod_base + cols, mask=mask, other=0.0).to(tl.float32)
        out = centered * inv_std
        out = out * (1.0 + scale) + shift
        tl.store(out_ptr + base + cols, out, mask=mask)


    @triton.jit
    def _gate_add_kernel(
        x_ptr,
        y_ptr,
        gate_ptr,
        rows_per_mod: tl.constexpr,
        dim: tl.constexpr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        cols = offsets % dim
        rows = offsets // dim
        mod_rows = rows // rows_per_mod
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        gate = tl.load(gate_ptr + mod_rows * dim + cols, mask=mask)
        tl.store(x_ptr + offsets, x + y * gate, mask=mask)


def _disable():
    global _TRITON_DISABLED
    _TRITON_DISABLED = True


def _eligible_tensor(x):
    return (
        triton is not None
        and not _TRITON_DISABLED
        and torch.is_tensor(x)
        and x.device.type == "cuda"
        and x.ndim == 3
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def layer_norm_modulate(x, shift, scale, eps):
    """Return fused LayerNorm(x) * (1 + scale) + shift, or None to fallback."""
    if not _eligible_tensor(x):
        return None
    if (
        not torch.is_tensor(shift)
        or not torch.is_tensor(scale)
        or shift.device != x.device
        or scale.device != x.device
        or shift.dtype != x.dtype
        or scale.dtype != x.dtype
        or shift.ndim != 3
        or scale.ndim != 3
        or shift.shape != scale.shape
        or int(shift.shape[0]) != int(x.shape[0])
        or int(shift.shape[1]) != 1
        or int(shift.shape[2]) != int(x.shape[2])
    ):
        return None
    dim = int(x.shape[-1])
    if dim <= 0 or dim > 65536:
        return None
    rows_per_mod = int(x.shape[1])
    rows = int(x.shape[0]) * rows_per_mod
    block = triton.next_power_of_2(dim)
    output = torch.empty_like(x)
    try:
        _layer_norm_modulate_kernel[(rows,)](
            x,
            shift,
            scale,
            output,
            rows_per_mod=rows_per_mod,
            dim=dim,
            eps=float(eps),
            BLOCK=block,
            num_warps=8 if block >= 2048 else 4,
        )
    except Exception:
        _disable()
        return None
    return output


def gate_add_inplace(x, y, gate):
    """Fuse x += y * gate for broadcast gate rows; return True on success."""
    if not _eligible_tensor(x):
        return False
    if (
        not torch.is_tensor(y)
        or not torch.is_tensor(gate)
        or y.shape != x.shape
        or y.device != x.device
        or y.dtype != x.dtype
        or not y.is_contiguous()
        or gate.device != x.device
        or gate.dtype != x.dtype
        or gate.ndim != 3
        or int(gate.shape[0]) != int(x.shape[0])
        or int(gate.shape[1]) != 1
        or int(gate.shape[2]) != int(x.shape[2])
    ):
        return False
    rows_per_mod = int(x.shape[1])
    dim = int(x.shape[2])
    total = int(x.numel())
    try:
        _gate_add_kernel[(triton.cdiv(total, 256),)](
            x,
            y,
            gate,
            rows_per_mod=rows_per_mod,
            dim=dim,
            total=total,
            BLOCK=256,
        )
    except Exception:
        _disable()
        return False
    return True
