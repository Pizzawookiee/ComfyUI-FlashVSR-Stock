"""Optional SpargeAttn executor for FlashVSR's logical LCSA mask."""

from __future__ import annotations

import math

import torch


SPARSE_BACKEND_OPTION = "flashvsr_sparse_attention_backend"


class FlashVSRSpargeBackend:
    """Execute FlashVSR's prescribed block mask with SpargeAttn.

    This is deliberately not a generic ComfyUI optimized-attention override.
    The FlashVSR runtime calls it only for streaming LCSA self-attention, while
    ModelAttentionBackend remains responsible for dense and cross-attention.
    """

    flashvsr_block_sparse = True

    def __init__(self):
        self.native_compact_disabled_reason = None
        self.reported_native_arch = None
        self.native_compact_validated = False

    @staticmethod
    def _kernel():
        try:
            from spas_sage_attn import block_sparse_sage2_attn_cuda
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "FlashVSR Sparge Attention requires a SpargeAttn wheel "
                "matching this ComfyUI Python, PyTorch, and CUDA build. "
                "Install spas_sage_attn from "
                "https://github.com/woct0rdho/SpargeAttn/releases and "
                "restart ComfyUI."
            ) from error
        return block_sparse_sage2_attn_cuda

    @staticmethod
    def _convert_mask_for_arch(mask: torch.Tensor, capability):
        """Convert logical 128x128 LCSA blocks for Sparge GPU geometry."""
        major, minor = capability
        if major < 8:
            raise RuntimeError(
                "FlashVSR Sparge Attention requires an Ampere-generation "
                "or newer NVIDIA GPU (compute capability 8.0 or newer)."
            )

        # Hopper consumes 64Q x 128K: split each logical query block. Other
        # supported architectures consume 128Q x 64K: split each logical key
        # block. Both physical halves inherit the one logical LCSA decision.
        if (major, minor) == (9, 0):
            mask = mask.repeat_interleave(2, dim=-2)
        else:
            mask = mask.repeat_interleave(2, dim=-1)
        return mask

    def _run_kernel(self, q, k, v, sparse_mask, profiler=None):
        """Run Sparge, exposing its internal stages when APIs are available."""
        kernel_marker = (
            profiler.profile_start(q) if profiler is not None else None
        )
        if profiler is None or not profiler.profile_enabled:
            output = self._kernel()(
                q, k, v, mask_id=sparse_mask, tensor_layout="HND"
            )
        else:
            output = self._run_profiled_kernel(q, k, v, sparse_mask, profiler)
        if profiler is not None:
            profiler.profile_end("sparge_kernel", kernel_marker)
        return output

    def _run_profiled_kernel(self, q, k, v, sparse_mask, profiler):
        """Mirror SpargeAttn's public HND path with CUDA-event boundaries."""
        try:
            import spas_sage_attn.core as core
            required = (
                "get_cuda_arch_versions", "get_vanilla_qk_quant",
                "block_map_lut_triton", "hyperparameter_check", "_fused",
            )
            if any(not hasattr(core, name) for name in required):
                raise AttributeError("SpargeAttn profiling APIs unavailable")
        except (ImportError, ModuleNotFoundError, AttributeError):
            return self._kernel()(
                q, k, v, mask_id=sparse_mask, tensor_layout="HND"
            )

        torch.cuda.set_device(v.device)
        marker = profiler.profile_start(q)
        if q.dtype in (torch.float32, torch.float16):
            q, k, v = (
                q.contiguous().to(torch.float16),
                k.contiguous().to(torch.float16),
                v.contiguous().to(torch.float16),
            )
        else:
            q, k, v = (
                q.contiguous().to(torch.bfloat16),
                k.contiguous().to(torch.bfloat16),
                v.contiguous().to(torch.float16),
            )
        profiler.profile_end("sparge_input_cast", marker)

        marker = profiler.profile_start(k)
        km = k.mean(dim=-2, keepdim=True)
        profiler.profile_end("sparge_k_smooth", marker)
        arch = core.get_cuda_arch_versions()[q.device.index]

        marker = profiler.profile_start(q)
        if arch == "sm90":
            q_int8, q_scale, k_int8, k_scale = (
                core.get_vanilla_qk_quant(q, k, km, 64, 128)
            )
        else:
            q_int8, q_scale, k_int8, k_scale = (
                core.get_vanilla_qk_quant(q, k, km, 128, 64)
            )
        profiler.profile_end("sparge_qk_quant", marker)

        marker = profiler.profile_start(sparse_mask)
        lut, valid_block_num = core.block_map_lut_triton(
            block_map=sparse_mask
        )
        profiler.profile_end("sparge_lut", marker)

        pv_threshold = core.hyperparameter_check(
            50, q.size(-3), q.device
        )
        scale = 1.0 / math.sqrt(q.size(-1))
        v_fp8 = None
        v_scale = None
        if arch in {"sm89", "sm90", "sm100", "sm120", "sm121"}:
            batch, kv_heads, kv_len, head_dim = v.shape
            padded_len = (kv_len + 127) // 128 * 128
            marker = profiler.profile_start(v)
            v_transposed = torch.empty(
                (batch, kv_heads, head_dim, padded_len),
                dtype=v.dtype, device=v.device,
            )
            core._fused.transpose_pad_permute_cuda(v, v_transposed, 1)
            profiler.profile_end("sparge_v_transpose", marker)

            marker = profiler.profile_start(v)
            v_fp8 = torch.empty(
                v_transposed.shape,
                dtype=torch.float8_e4m3fn,
                device=v.device,
            )
            v_scale = torch.empty(
                (batch, kv_heads, head_dim),
                dtype=torch.float32, device=v.device,
            )
            maximum = 448.0 if arch == "sm90" else 2.25
            core._fused.scale_fuse_quant_cuda(
                v_transposed, v_fp8, v_scale, kv_len, maximum, 1
            )
            profiler.profile_end("sparge_v_quant", marker)

        output = torch.empty_like(q)
        marker = profiler.profile_start(q)
        if arch in {"sm80", "sm86", "sm87"}:
            core._qattn_sm80.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
                q_int8, k_int8, v, output, lut, valid_block_num,
                pv_threshold, q_scale, k_scale, 1, False, 1, scale, 0,
            )
        elif arch in {"sm89", "sm100", "sm120", "sm121"}:
            if core.get_cuda_version() < (12, 8):
                function = core._qattn_sm89.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
            else:
                function = core._qattn_sm89.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
            function(
                q_int8, k_int8, v_fp8, output, lut, valid_block_num,
                pv_threshold, q_scale, k_scale, v_scale,
                1, False, 1, scale, 0,
            )
        elif arch == "sm90":
            core._qattn_sm90.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(
                q_int8, k_int8, v_fp8, output, lut, valid_block_num,
                pv_threshold, q_scale, k_scale, v_scale,
                1, False, 1, scale, 0,
            )
        else:
            raise ValueError(f"Unsupported CUDA architecture: {arch}")
        profiler.profile_end("sparge_attention_cuda", marker)
        return output

    @staticmethod
    def _restore(output_hnd, q, heads, q_block_to_original,
                 profiler=None):
        batch, q_tokens, channels = q.shape
        head_dim = channels // heads
        restore_marker = (
            profiler.profile_start(q) if profiler is not None else None
        )
        attended = torch.empty_like(q)
        output_nhd = output_hnd.permute(0, 2, 1, 3)
        if output_nhd.dtype != q.dtype:
            output_nhd = output_nhd.to(dtype=q.dtype)
        attended.view(batch, q_tokens, heads, head_dim).index_copy_(
            1, q_block_to_original, output_nhd
        )
        if profiler is not None:
            profiler.profile_end("sparge_restore", restore_marker)
        return attended

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
        """Run one complete sparse layer and restore stock Wan token order."""
        if q.device.type != "cuda":
            raise RuntimeError(
                "FlashVSR Sparge Attention requires a CUDA device."
            )
        batch, q_tokens, channels = q.shape
        if channels % heads:
            raise RuntimeError(
                "FlashVSR Sparge Attention received a channel count that is "
                "not divisible by the attention head count."
            )
        if (
            k.shape != v.shape
            or k.shape[0] != q.shape[0]
            or k.shape[2] != channels
        ):
            raise RuntimeError(
                "FlashVSR Sparge Attention received incompatible Q/K/V "
                "batch or channel shapes."
            )
        head_dim = channels // heads
        if head_dim not in (64, 128):
            raise RuntimeError(
                "FlashVSR Sparge Attention supports 64- or 128-wide "
                f"attention heads, but received {head_dim}."
            )
        k_tokens = k.shape[1]
        if q_tokens % 128 or k_tokens % 128:
            raise RuntimeError(
                "FlashVSR logical LCSA requires Q and K token counts aligned "
                "to 128-token blocks."
            )

        expected_mask = (
            batch,
            heads,
            q_tokens // 128,
            k_tokens // 128,
        )
        if tuple(mask.shape) != expected_mask:
            raise RuntimeError(
                "FlashVSR supplied an unexpected LCSA block mask shape: "
                f"{tuple(mask.shape)} != {expected_mask}."
            )

        # Gather directly from stock Wan order into the final contiguous HND
        # kernel layout. The previous route first materialized full [B,N,C]
        # block-order tensors and then copied all three again while transposing.
        layout_marker = (
            profiler.profile_start(q) if profiler is not None else None
        )
        q_hnd = (
            q.view(batch, q_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, q_block_to_original)
            .contiguous()
        )
        k_hnd = (
            k.view(batch, k_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, k_block_to_original)
            .contiguous()
        )
        v_hnd = (
            v.view(batch, k_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, k_block_to_original)
            .contiguous()
        )
        if profiler is not None:
            profiler.profile_end("sparge_layout_qkv", layout_marker)

        mask_marker = (
            profiler.profile_start(q) if profiler is not None else None
        )
        capability = torch.cuda.get_device_capability(q.device)
        sparse_mask = self._convert_mask_for_arch(mask, capability)
        sparse_mask = sparse_mask.to(
            device=q.device, dtype=torch.int8
        ).contiguous()
        if profiler is not None:
            profiler.profile_end("sparge_mask_convert", mask_marker)

        output_hnd = self._run_kernel(
            q_hnd, k_hnd, v_hnd, sparse_mask, profiler
        )

        # Scatter into the final stock-Wan layout without first creating a
        # complete contiguous NHD output copy. Release all bounded full-layer
        # temporaries before returning to Wan's output projection.
        attended = self._restore(
            output_hnd, q, heads, q_block_to_original, profiler
        )
        del output_hnd, sparse_mask, q_hnd, k_hnd, v_hnd
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
        """Run sparse attention without constructing FP16 [B,N,C] cache."""
        if q.device.type != "cuda":
            raise RuntimeError(
                "FlashVSR Sparge Attention requires a CUDA device."
            )
        batch, q_tokens, channels = q.shape
        if channels % heads:
            raise RuntimeError("Invalid compact-cache attention head count.")
        head_dim = channels // heads
        if head_dim not in (64, 128) or q_tokens % 128:
            raise RuntimeError("Invalid compact-cache Sparge Q geometry.")
        total_tokens = (
            descriptor.history_frames * descriptor.tokens_per_frame
            + descriptor.current_k.shape[1]
        )
        expected_mask = (
            batch, heads, q_tokens // 128, total_tokens // 128
        )
        if tuple(mask.shape) != expected_mask:
            raise RuntimeError(
                "FlashVSR supplied an unexpected compact LCSA mask shape: "
                f"{tuple(mask.shape)} != {expected_mask}."
            )

        layout_marker = (
            profiler.profile_start(q) if profiler is not None else None
        )
        q_hnd = (
            q.view(batch, q_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, q_block_to_original)
            .contiguous()
        )
        if profiler is not None:
            profiler.profile_end("sparge_layout_qkv", layout_marker)
        mask_marker = (
            profiler.profile_start(q) if profiler is not None else None
        )
        capability = torch.cuda.get_device_capability(q.device)
        sparse_mask = self._convert_mask_for_arch(mask, capability).to(
            device=q.device, dtype=torch.int8
        ).contiguous()
        if profiler is not None:
            profiler.profile_end("sparge_mask_convert", mask_marker)

        output_hnd = None
        if self.native_compact_disabled_reason is None:
            try:
                native_marker = (
                    profiler.profile_start(q_hnd) if profiler else None
                )
                output_hnd = self._run_native_compact(
                    q_hnd,
                    descriptor,
                    cache,
                    slot_block_to_original,
                    sparse_mask,
                    profiler,
                )
                if profiler:
                    profiler.profile_end("sparge_kernel", native_marker)
            except Exception as error:
                self.native_compact_disabled_reason = str(error)
                print(
                    "[FlashVSR] native compact Sparge adapter unavailable: "
                    f"{error}; using the v0.33 compatibility path."
                )
        if output_hnd is None:
            k_hnd, v_hnd = cache.materialize_hnd(
                descriptor, slot_block_to_original, profiler
            )
            output_hnd = self._run_kernel(
                q_hnd, k_hnd, v_hnd, sparse_mask, profiler
            )
            del k_hnd, v_hnd
        attended = self._restore(
            output_hnd, q, heads, q_block_to_original, profiler
        )
        del output_hnd, sparse_mask, q_hnd
        return attended

    def _run_native_compact(
        self,
        q_hnd,
        descriptor,
        cache,
        slot_block_to_original,
        sparse_mask,
        profiler,
    ):
        """Consume row-INT8 history through Sparge's compatible lower ABI."""
        try:
            import spas_sage_attn.core as core
            import spas_sage_attn.utils as sparge_utils
            from .native_sparge import native_k, native_v_fp8
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "native Sparge dependencies are unavailable"
            ) from error

        required = (
            "get_cuda_arch_versions", "block_map_lut_triton",
            "hyperparameter_check",
        )
        if any(not hasattr(core, name) for name in required):
            raise RuntimeError("installed SpargeAttn lower API is incompatible")
        if not hasattr(sparge_utils, "get_quant"):
            raise RuntimeError("installed SpargeAttn lacks get_quant")

        torch.cuda.set_device(q_hnd.device)
        arch = core.get_cuda_arch_versions()[q_hnd.device.index]
        supported = {
            "sm80", "sm86", "sm87", "sm89", "sm90",
            "sm100", "sm120", "sm121",
        }
        if arch not in supported:
            raise RuntimeError(f"unsupported native Sparge architecture {arch}")
        if self.reported_native_arch != arch:
            status = (
                "RTX-4000 target" if arch == "sm89"
                else "experimental, non-RTX-4000 path untested"
            )
            print(
                f"[FlashVSR] native compact Sparge adapter={arch} "
                f"({status})."
            )
            self.reported_native_arch = arch

        if q_hnd.dtype in (torch.float32, torch.float16):
            q_hnd = q_hnd.contiguous().to(torch.float16)
        else:
            q_hnd = q_hnd.contiguous().to(torch.bfloat16)
        block_q, block_k = ((64, 128) if arch == "sm90" else (128, 64))

        transfer_marker = profiler.profile_start(q_hnd) if profiler else None
        with cache.acquire_compact_slots(
            descriptor, q_hnd.device
        ) as acquired:
            if profiler:
                profiler.profile_end(
                    "kv_cache_compact_h2d", transfer_marker
                )
            q_marker = profiler.profile_start(q_hnd) if profiler else None
            q_int8, q_scale = sparge_utils.get_quant(
                q_hnd, None, block_q
            )
            if profiler:
                profiler.profile_end("sparge_q_quant", q_marker)
            k_int8, k_scale = native_k(
                descriptor,
                acquired,
                slot_block_to_original,
                block_k,
                profiler,
            )
            if arch in {"sm89", "sm90", "sm100", "sm120", "sm121"}:
                v_kernel, v_scale = native_v_fp8(
                    descriptor,
                    acquired,
                    slot_block_to_original,
                    arch,
                    profiler,
                )
            else:
                # Ampere Sparge consumes FP16/BF16 V. Native K still avoids
                # reconstructing K, while v0.33's direct final-HND V route is
                # retained for the architecture's lower kernel.
                v_kernel = cache.materialize_v_hnd(
                    descriptor, slot_block_to_original, profiler
                ).to(torch.float16)
                v_scale = None

            if not self.native_compact_validated:
                self._validate_native_compact(
                    descriptor,
                    acquired,
                    slot_block_to_original,
                    k_int8,
                    k_scale,
                    v_kernel,
                    v_scale,
                    block_k,
                    arch,
                )
                self.native_compact_validated = True
                print(
                    "[FlashVSR] native compact Sparge converter validation "
                    "passed."
                )

            marker = profiler.profile_start(sparse_mask) if profiler else None
            lut, valid_block_num = core.block_map_lut_triton(
                block_map=sparse_mask
            )
            if profiler:
                profiler.profile_end("sparge_lut", marker)
            pv_threshold = core.hyperparameter_check(
                50, q_hnd.size(-3), q_hnd.device
            )
            output = torch.empty_like(q_hnd)
            scale = 1.0 / math.sqrt(q_hnd.size(-1))
            marker = profiler.profile_start(q_hnd) if profiler else None
            if arch in {"sm80", "sm86", "sm87"}:
                core._qattn_sm80.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
                    q_int8, k_int8, v_kernel, output,
                    lut, valid_block_num, pv_threshold,
                    q_scale, k_scale, 1, False, 1, scale, 0,
                )
            elif arch in {"sm89", "sm100", "sm120", "sm121"}:
                if core.get_cuda_version() < (12, 8):
                    function = core._qattn_sm89.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
                else:
                    function = core._qattn_sm89.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
                function(
                    q_int8, k_int8, v_kernel, output,
                    lut, valid_block_num, pv_threshold,
                    q_scale, k_scale, v_scale,
                    1, False, 1, scale, 0,
                )
            else:
                core._qattn_sm90.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(
                    q_int8, k_int8, v_kernel, output,
                    lut, valid_block_num, pv_threshold,
                    q_scale, k_scale, v_scale,
                    1, False, 1, scale, 0,
                )
            if profiler:
                profiler.profile_end("sparge_attention_cuda", marker)
            return output

    @staticmethod
    def _validate_native_compact(
        descriptor,
        acquired,
        token_map,
        k_int8,
        k_scale,
        v_kernel,
        v_scale,
        block_k,
        arch,
    ):
        """Validate small real-cache samples without full compatibility buffers."""
        cached_k, cached_v = acquired[0]
        batch, slot_tokens, channels = cached_k.shape
        heads = descriptor.heads
        head_dim = channels // heads
        sample_k = min(block_k, slot_tokens)
        indices_k = token_map[:sample_k]
        cached_k_values = (
            cached_k.qdata.view(batch, slot_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, indices_k)
            .float()
        )
        cached_k_scales = (
            cached_k.scale.view(batch, slot_tokens, heads, 1)
            .permute(0, 2, 1, 3)
            .index_select(2, indices_k)
        )
        reference = descriptor.k_reference_mean.to(
            device=k_int8.device, dtype=torch.float32
        ).unsqueeze(2)
        expected_k = cached_k_values.mul(cached_k_scales).sub(reference)
        reconstructed_k = k_int8[:, :, :sample_k].float().mul(
            k_scale[:, :, :1].unsqueeze(-1)
        )
        denominator = expected_k.norm().clamp_min(1.0e-6)
        k_error = ((reconstructed_k - expected_k).norm() / denominator).item()
        if not math.isfinite(k_error) or k_error > 0.20:
            raise RuntimeError(
                f"native K converter validation error was {k_error:.4f}"
            )

        if arch not in {"sm89", "sm90", "sm100", "sm120", "sm121"}:
            return
        sample_v = min(16, slot_tokens)
        indices_v = token_map[:sample_v]
        expected_v = (
            cached_v.qdata.view(batch, slot_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, indices_v)
            .float()
        )
        expected_v.mul_(
            cached_v.scale.view(batch, slot_tokens, heads, 1)
            .permute(0, 2, 1, 3)
            .index_select(2, indices_v)
        )
        logical = torch.arange(sample_v, device=k_int8.device)
        within = logical.remainder(16)
        permuted = (
            logical.div(16, rounding_mode="floor") * 16
            + within.div(8, rounding_mode="floor") * 2
            + within.div(2, rounding_mode="floor").remainder(4) * 4
            + within.remainder(2)
        )
        reconstructed_v = (
            v_kernel[:, :, :, permuted]
            .float()
            .permute(0, 1, 3, 2)
            .mul(v_scale.unsqueeze(2))
        )
        denominator = expected_v.norm().clamp_min(1.0e-6)
        v_error = ((reconstructed_v - expected_v).norm() / denominator).item()
        if not math.isfinite(v_error) or v_error > 0.20:
            raise RuntimeError(
                f"native V converter validation error was {v_error:.4f}"
            )
        current = descriptor.current_v
        current_tokens = current.shape[1]
        current_sample = min(16, current_tokens)
        current_indices = token_map[:current_sample]
        expected_current = (
            current.view(batch, current_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, current_indices)
            .float()
        )
        current_logical = torch.arange(
            current_sample, device=k_int8.device
        )
        within = current_logical.remainder(16)
        current_permuted = (
            current_logical.div(16, rounding_mode="floor") * 16
            + within.div(8, rounding_mode="floor") * 2
            + within.div(2, rounding_mode="floor").remainder(4) * 4
            + within.remainder(2)
        )
        history_tokens = descriptor.history_frames * descriptor.tokens_per_frame
        reconstructed_current = (
            v_kernel[:, :, :, history_tokens + current_permuted]
            .float()
            .permute(0, 1, 3, 2)
            .mul(v_scale.unsqueeze(2))
        )
        denominator = expected_current.norm().clamp_min(1.0e-6)
        current_error = (
            (reconstructed_current - expected_current).norm() / denominator
        ).item()
        if not math.isfinite(current_error) or current_error > 0.20:
            raise RuntimeError(
                "native current-V converter validation error was "
                f"{current_error:.4f}"
            )


def apply_sparge_backend(model):
    """Attach Sparge as FlashVSR's private sparse route, preserving fallback."""
    patched = model.clone()
    transformer_options = patched.model_options.setdefault(
        "transformer_options", {}
    )
    backend = FlashVSRSpargeBackend()
    transformer_options[SPARSE_BACKEND_OPTION] = backend

    # When Configure FlashVSR has already installed its dispatcher, update the
    # live runtime as well. If this patch is applied first, patch_model reads
    # the private option later. This makes workflow patch order immaterial.
    override = transformer_options.get("optimized_attention_override")
    runtime = getattr(override, "flashvsr_runtime", None)
    if runtime is not None:
        runtime.set_sparse_attention_backend(backend)
    return patched


# Preserve internal import compatibility with 0.17.x scripts. These aliases
# now attach the private Sparge route instead of replacing ModelAttentionBackend.
FlashVSRBlockSparseBackend = FlashVSRSpargeBackend
FlashVSRSparseSageBackend = FlashVSRSpargeBackend
apply_block_sparse_backend = apply_sparge_backend
apply_sparse_sage_backend = apply_sparge_backend
