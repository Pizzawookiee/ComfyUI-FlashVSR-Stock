"""Optional SpargeAttn executor for FlashVSR's native LCSA mask."""

from __future__ import annotations

import torch


SPARSE_BACKEND_OPTION = "flashvsr_sparse_attention_backend"


class FlashVSRSpargeBackend:
    """Execute FlashVSR's prescribed block mask with SpargeAttn.

    This is deliberately not a generic ComfyUI optimized-attention override.
    The FlashVSR runtime calls it only for streaming LCSA self-attention, while
    ModelAttentionBackend remains responsible for dense and cross-attention.
    """

    flashvsr_block_sparse = True

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
        """Convert native 128x64 LCSA blocks for Sparge's GPU geometry."""
        major, minor = capability
        if major < 8:
            raise RuntimeError(
                "FlashVSR Sparge Attention requires an Ampere-generation "
                "or newer NVIDIA GPU (compute capability 8.0 or newer)."
            )

        # Sparge's Hopper kernel consumes 64-query x 128-key blocks. Split
        # each LCSA query block and conservatively merge each adjacent key
        # pair so no connection selected by FlashVSR is discarded.
        if (major, minor) == (9, 0):
            if mask.shape[-1] % 2:
                mask = torch.nn.functional.pad(mask, (0, 1), value=False)
            mask = mask.repeat_interleave(2, dim=-2)
            mask = mask.reshape(*mask.shape[:-1], -1, 2).any(dim=-1)
        return mask

    def run_flashvsr(
        self,
        q,
        k,
        v,
        heads,
        mask,
        block_to_original,
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
        if k.shape != q.shape or v.shape != q.shape:
            raise RuntimeError(
                "FlashVSR Sparge Attention requires equal self-attention "
                "Q/K/V shapes."
            )
        head_dim = channels // heads
        if head_dim not in (64, 128):
            raise RuntimeError(
                "FlashVSR Sparge Attention supports 64- or 128-wide "
                f"attention heads, but received {head_dim}."
            )
        if q_tokens % 128 or k.shape[1] % 64:
            raise RuntimeError(
                "FlashVSR Sparge Attention requires 128-query by 64-key "
                "aligned token counts."
            )

        expected_mask = (
            batch,
            heads,
            q_tokens // 128,
            k.shape[1] // 64,
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
            .index_select(2, block_to_original)
            .contiguous()
        )
        k_hnd = (
            k.view(batch, q_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, block_to_original)
            .contiguous()
        )
        v_hnd = (
            v.view(batch, q_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, block_to_original)
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

        kernel_marker = (
            profiler.profile_start(q) if profiler is not None else None
        )
        output_hnd = self._kernel()(
            q_hnd,
            k_hnd,
            v_hnd,
            mask_id=sparse_mask,
            tensor_layout="HND",
        )
        if profiler is not None:
            profiler.profile_end("sparge_kernel", kernel_marker)

        # Scatter into the final stock-Wan layout without first creating a
        # complete contiguous NHD output copy. Release all bounded full-layer
        # temporaries before returning to Wan's output projection.
        restore_marker = (
            profiler.profile_start(q) if profiler is not None else None
        )
        attended = torch.empty_like(q)
        output_nhd = output_hnd.permute(0, 2, 1, 3)
        if output_nhd.dtype != q.dtype:
            # Sparge computes FP32 inputs in FP16. Restore Wan's activation
            # dtype before its output projection; FP16/BF16 paths are views.
            output_nhd = output_nhd.to(dtype=q.dtype)
        attended.view(batch, q_tokens, heads, head_dim).index_copy_(
            1,
            block_to_original,
            output_nhd,
        )
        if profiler is not None:
            profiler.profile_end("sparge_restore", restore_marker)
        del output_nhd, output_hnd, sparse_mask, q_hnd, k_hnd, v_hnd
        return attended


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
