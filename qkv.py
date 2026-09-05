"""Wan streaming QKV helpers with guarded ConvRot INT8 acceleration.

This module deliberately keeps Wan's Q, K and V modules separate.  ComfyUI
therefore remains responsible for their normal weight lifetime, including
ModelPatcherDynamic/AIMDO leases.  When all three projections use compatible
TensorWise INT8 ConvRot weights, their input is rotated and quantized once and
the resulting activation carrier is reused by the three projections.

The implementation is original to this project.  It uses ComfyUI and
Comfy-Kitchen runtime interfaces, and does not import another custom node.
Every accelerated operation is capability guarded and has the stock Wan
forward as its correctness fallback.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ConvRotQKVFormat:
    group_size: int
    hidden: int
    output: int
    output_dtype: torch.dtype


class SharedQKVUnavailable(RuntimeError):
    """Expected capability miss for the optional shared-QKV path."""


def _weight_functions(module):
    return tuple(getattr(module, "weight_function", ()) or ())


def _bias_functions(module):
    return tuple(getattr(module, "bias_function", ()) or ())


def _describe_convrot_linear(module):
    weight = getattr(module, "weight", None)
    params = getattr(weight, "_params", None)
    layout = (
        getattr(weight, "_layout_cls", None)
        or getattr(module, "layout_type", None)
    )
    if layout != "TensorWiseINT8Layout":
        return None
    if params is None:
        return None
    if bool(getattr(params, "transposed", False)):
        return None
    if not bool(getattr(params, "convrot", False)):
        return None
    group_size = int(getattr(params, "convrot_groupsize", 0) or 0)
    shape = tuple(int(value) for value in getattr(weight, "shape", ()))
    if len(shape) != 2 or group_size <= 0 or shape[1] % group_size:
        return None
    if _weight_functions(module) or _bias_functions(module):
        # Weight functions include LoRA and other patches.  Falling back is
        # mandatory: bypassing them would silently change model behavior.
        return None
    return ConvRotQKVFormat(
        group_size=group_size,
        hidden=shape[1],
        output=shape[0],
        output_dtype=getattr(params, "orig_dtype", torch.bfloat16),
    )


def inspect_self_attention(module):
    """Return the common ConvRot format or ``None`` for stock fallback."""
    formats = tuple(
        _describe_convrot_linear(getattr(module, name, None))
        for name in ("q", "k", "v")
    )
    if any(item is None for item in formats):
        return None
    first = formats[0]
    if any(
        item.group_size != first.group_size
        or item.hidden != first.hidden
        or item.output != first.output
        for item in formats[1:]
    ):
        return None
    if first.hidden != int(getattr(module, "dim", first.hidden)):
        return None
    return first


def inventory_wan_blocks(blocks):
    formats = []
    for block in blocks:
        self_attention = getattr(block, "self_attn", None)
        current = inspect_self_attention(self_attention)
        if current is None:
            return None
        formats.append(current)
    if not formats:
        return None
    first = formats[0]
    if any(item != first for item in formats[1:]):
        return None
    return first


def _plain_int8_weight(weight):
    try:
        from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout
    except (ImportError, ModuleNotFoundError) as error:
        raise SharedQKVUnavailable(
            "ComfyUI TensorWise INT8 support is unavailable"
        ) from error
    if not isinstance(weight, QuantizedTensor):
        raise SharedQKVUnavailable("projection weight was not quantized")
    if getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
        raise SharedQKVUnavailable("projection weight was not TensorWise INT8")
    params = getattr(weight, "_params", None)
    if params is None or bool(getattr(params, "transposed", False)):
        raise SharedQKVUnavailable("projection weight layout was unsupported")
    if not bool(getattr(params, "convrot", False)):
        raise SharedQKVUnavailable("projection weight did not use ConvRot")
    qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    return qdata, scale, params


def _quantize_shared_input(x, group_size):
    if x.device.type != "cuda" or x.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise SharedQKVUnavailable(
            "shared ConvRot QKV requires CUDA FP16 or BF16 activations"
        )
    try:
        from comfy_kitchen.backends import cuda as kitchen_cuda
    except (ImportError, ModuleNotFoundError) as error:
        raise SharedQKVUnavailable(
            "the Comfy-Kitchen CUDA backend is unavailable"
        ) from error
    quantizer = getattr(
        kitchen_cuda, "quantize_int8_rowwise_convrot64", None
    )
    if not callable(quantizer):
        raise SharedQKVUnavailable(
            "Comfy-Kitchen lacks the ConvRot INT8 activation quantizer"
        )
    flattened = x.reshape(-1, x.shape[-1]).contiguous()
    qdata, scale = quantizer(flattened, int(group_size))
    return qdata, scale.reshape(-1, 1).contiguous()


def _cutlass_project(x_qdata, x_scale, weight, bias, output_dtype):
    """Consume a prequantized activation through Comfy-Kitchen's CUDA ABI.

    Comfy-Kitchen currently exposes its fused INT8 projection through the
    CUDA backend extension used by ``int8_linear``.  The adapter is guarded
    by exact symbol checks, so API changes select stock Wan rather than
    producing an approximate result.
    """
    try:
        from comfy_kitchen.backends import cuda as kitchen_cuda
    except (ImportError, ModuleNotFoundError) as error:
        raise SharedQKVUnavailable("Comfy-Kitchen CUDA is unavailable") from error

    extension = getattr(kitchen_cuda, "_C", None)
    wrap = getattr(kitchen_cuda, "_wrap_for_dlpack", None)
    dtype_codes = getattr(kitchen_cuda, "DTYPE_TO_CODE", None)
    kernel = getattr(extension, "cutlass_int8_dequant", None)
    if (
        extension is None
        or not callable(wrap)
        or not isinstance(dtype_codes, dict)
        or not callable(kernel)
        or output_dtype not in dtype_codes
    ):
        raise SharedQKVUnavailable(
            "the installed Comfy-Kitchen does not expose its prepared INT8 "
            "projection ABI"
        )

    weight_qdata, weight_scale, params = _plain_int8_weight(weight)
    if int(getattr(params, "convrot_groupsize", 0)) <= 0:
        raise SharedQKVUnavailable("invalid ConvRot group size")
    if weight_qdata.device != x_qdata.device:
        raise SharedQKVUnavailable("projection weight was not on the input device")
    major, _minor = torch.cuda.get_device_capability(x_qdata.device)
    if major < 8:
        raise SharedQKVUnavailable(
            "the prepared CUTLASS projection requires an SM80-or-newer GPU"
        )

    rows = x_qdata.shape[0]
    output_features = weight_qdata.shape[0]
    output = torch.empty(
        (rows, output_features),
        device=x_qdata.device,
        dtype=output_dtype,
    )
    weight_scale = weight_scale.to(
        device=x_qdata.device, dtype=torch.float32
    ).reshape(-1).contiguous()
    if weight_scale.numel() not in (1, output_features):
        raise SharedQKVUnavailable("unsupported ConvRot weight scale shape")
    if weight_scale.numel() == 1:
        weight_scale = weight_scale.expand(output_features).contiguous()
    if bias is None:
        bias_arg = torch.empty(
            (0,), device=x_qdata.device, dtype=torch.float32
        )
    else:
        # cutlass_int8_dequant reads bias through a float pointer regardless
        # of the output dtype. Passing FP16/BF16 here silently reinterprets
        # pairs of half values as FP32 and produces structured corruption.
        bias_arg = bias.to(
            device=x_qdata.device, dtype=torch.float32
        ).contiguous()

    try:
        used = kernel(
            wrap(x_qdata),
            wrap(weight_qdata.contiguous()),
            wrap(x_scale),
            wrap(weight_scale),
            wrap(bias_arg),
            wrap(output),
            dtype_codes[output_dtype],
            torch.cuda.current_stream(x_qdata.device).cuda_stream,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise SharedQKVUnavailable(
            f"prepared INT8 projection was rejected: {error}"
        ) from error
    if not used:
        raise SharedQKVUnavailable(
            "Comfy-Kitchen declined the prepared INT8 projection shape"
        )
    return output


def _projection_context(stack, linear, x):
    import comfy.ops

    return stack.enter_context(
        comfy.ops.CastBiasWeightContext(
            linear,
            x,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=True,
        )
    )


def _resident_projection(linear, x):
    if _weight_functions(linear) or _bias_functions(linear):
        return None
    weight = getattr(linear, "weight", None)
    try:
        qdata, _scale, _params = _plain_int8_weight(weight)
    except SharedQKVUnavailable:
        return None
    if (
        hasattr(linear, "_v")
        or bool(getattr(linear, "comfy_cast_weights", False))
        or qdata.device != x.device
    ):
        return None
    bias = getattr(linear, "bias", None)
    if bias is not None and bias.device != x.device:
        return None
    return weight, bias


def _project_shared(module, x, runtime, format_info):
    import comfy.ops

    # Match the stock Linear forward contract before touching dynamically
    # managed weights or reusable offload buffers.
    comfy.ops.run_every_op()
    quant_marker = runtime.profile_start(x)
    x_qdata, x_scale = _quantize_shared_input(x, format_info.group_size)
    runtime.profile_end("qkv_convrot_quant", quant_marker)

    resident = tuple(
        _resident_projection(getattr(module, name), x)
        for name in ("q", "k", "v")
    )
    resident_group = all(item is not None for item in resident)
    outputs = []
    if resident_group:
        # Three separate stock parameters, one shared carrier, and no dynamic
        # leases.  The calls are grouped under a single lifetime even though
        # current Comfy-Kitchen releases them as three CUTLASS launches.
        for name, (weight, bias) in zip(("q", "k", "v"), resident):
            marker = runtime.profile_start(x_qdata)
            outputs.append(
                _cutlass_project(
                    x_qdata, x_scale, weight, bias, x.dtype
                )
            )
            runtime.profile_end(f"qkv_{name}_gemm", marker)
        path = "resident_shared_int8"
    else:
        # Dynamic VRAM path: each weight lease ends before the next begins.
        # The shared activation carrier is the only persistent QKV input.
        for name in ("q", "k", "v"):
            linear = getattr(module, name)
            marker = runtime.profile_start(x_qdata)
            with ExitStack() as stack:
                weight, bias = _projection_context(stack, linear, x)
                projected = _cutlass_project(
                    x_qdata, x_scale, weight, bias, x.dtype
                )
            outputs.append(projected)
            runtime.profile_end(f"qkv_{name}_gemm", marker)
        path = "sequential_dynamic_int8"

    shape = (*x.shape[:-1], format_info.output)
    q, k, v = (value.reshape(shape) for value in outputs)
    runtime.note_qkv_path(path)
    return q, k, v


class WanStreamingQKVForward:
    """Callable object patch for stock Wan self-attention."""

    def __init__(self, module, original_forward, runtime, format_info):
        self.module = module
        self.original_forward = original_forward
        self.runtime = runtime
        self.format_info = format_info

    def __call__(self, x, freqs, transformer_options=None):
        options = transformer_options or {}
        runtime = options.get("flashvsr_active_runtime", self.runtime)
        if (
            runtime is None
            or not getattr(runtime, "streaming_active", False)
            or getattr(runtime, "qkv_projection_mode", "stock")
            != "shared_int8_experimental"
            or not getattr(runtime, "int8_qkv_capable", False)
        ):
            return self.original_forward(
                x, freqs, transformer_options=options
            )

        try:
            q, k, v = _project_shared(
                self.module, x, runtime, self.format_info
            )
        except SharedQKVUnavailable as error:
            runtime.disable_int8_qkv(str(error))
            return self.original_forward(
                x, freqs, transformer_options=options
            )

        from comfy.ldm.flux.math import apply_rope1
        from comfy.ldm.modules.attention import optimized_attention

        batch, sequence = x.shape[:2]
        heads = int(self.module.num_heads)
        head_dim = int(self.module.head_dim)
        q_marker = runtime.profile_start(q)
        q = self.module.norm_q(q).view(batch, sequence, heads, head_dim)
        q = apply_rope1(q, freqs)
        runtime.profile_end("qkv_q_norm_rope", q_marker)
        k_marker = runtime.profile_start(k)
        k = self.module.norm_k(k).view(batch, sequence, heads, head_dim)
        k = apply_rope1(k, freqs)
        runtime.profile_end("qkv_k_norm_rope", k_marker)

        attended = optimized_attention(
            q.view(batch, sequence, heads * head_dim),
            k.view(batch, sequence, heads * head_dim),
            v.view(batch, sequence, heads * head_dim),
            heads=heads,
            transformer_options=options,
        )

        patches = options.get("patches", {})
        for patch in patches.get("attn1_patch", ()):
            attended = patch({
                "x": attended,
                "q": q,
                "k": k,
                "transformer_options": options,
            })
        return self.module.o(attended)


def install_wan_qkv_patches(patcher, runtime, blocks):
    """Install guarded forward patches and return common capability info."""
    common = inventory_wan_blocks(blocks)
    runtime.configure_int8_qkv(common)
    if common is None:
        return None
    for index, block in enumerate(blocks):
        module = block.self_attn
        wrapper = WanStreamingQKVForward(
            module,
            module.forward,
            runtime,
            inspect_self_attention(module),
        )
        patcher.add_object_patch(
            f"diffusion_model.blocks.{index}.self_attn.forward",
            wrapper,
        )
    return common


class Int8Carrier:
    """Compact per-token/per-head INT8 cache with an FP32 row scale."""

    __slots__ = ("qdata", "scale", "shape", "dtype", "head_dim")

    def __init__(self, qdata, scale, shape, dtype, head_dim):
        self.qdata = qdata
        self.scale = scale
        self.shape = tuple(int(value) for value in shape)
        self.dtype = dtype
        self.head_dim = int(head_dim)

    @property
    def device(self):
        return self.qdata.device

    @property
    def nbytes(self):
        return (
            self.qdata.numel() * self.qdata.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    @classmethod
    def from_tensor(cls, source, device, heads):
        heads = max(1, int(heads))
        if source.shape[-1] % heads:
            raise ValueError(
                "FlashVSR INT8 cache channels must be divisible by heads."
            )
        head_dim = source.shape[-1] // heads
        # Scaling each attention head independently avoids allowing one
        # outlier head to consume the precision of every other head.
        source_2d = source.detach().reshape(-1, head_dim)
        qdata = torch.empty_like(source_2d, dtype=torch.int8)
        scale = torch.empty(
            (source_2d.shape[0], 1),
            device=source.device,
            dtype=torch.float32,
        )
        # Keep temporary FP32 work bounded.  Quantizing a complete HD token
        # tensor at once can otherwise consume more memory than the cache is
        # intended to save.
        bytes_per_row = max(1, source_2d.shape[-1] * 8)
        rows_per_chunk = max(1, (32 * 1024 * 1024) // bytes_per_row)
        for start in range(0, source_2d.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, source_2d.shape[0])
            current = source_2d[start:end]
            current_scale = current.float().abs_().amax(
                dim=-1, keepdim=True
            )
            current_scale.div_(127.0).clamp_min_(1.0e-8)
            working = current.float().div_(current_scale)
            working.round_().clamp_(-127, 127)
            qdata[start:end].copy_(working.to(dtype=torch.int8))
            scale[start:end].copy_(current_scale)
        target = torch.device(device)
        if qdata.device != target:
            qdata = qdata.to(device=target)
            scale = scale.to(device=target)
        return cls(
            qdata.contiguous(),
            scale.to(dtype=torch.float32).contiguous(),
            source.shape,
            source.dtype,
            head_dim,
        )

    def copy_to(self, destination):
        flat = destination.reshape(-1, self.head_dim)
        if tuple(destination.shape) != self.shape:
            raise ValueError(
                "FlashVSR carrier destination shape does not match cache"
            )
        # Bound dequantization temporaries.  A direct carrier-consuming sparse
        # kernel can replace this adapter later without changing cache state.
        bytes_per_row = max(1, destination.shape[-1] * 5)
        rows_per_chunk = max(1, (32 * 1024 * 1024) // bytes_per_row)
        for start in range(0, flat.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, flat.shape[0])
            qdata = self.qdata[start:end].to(
                device=destination.device,
                dtype=destination.dtype,
            )
            scale = self.scale[start:end].to(
                device=destination.device,
                dtype=destination.dtype,
            )
            flat[start:end].copy_(qdata.mul_(scale))


def carrier_nbytes(shape, heads):
    rows = math.prod(shape[:-1]) * int(heads)
    values = math.prod(shape)
    return values + rows * 4
