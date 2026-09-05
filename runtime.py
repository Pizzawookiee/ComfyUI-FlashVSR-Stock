from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import math

import torch

import comfy.ldm.modules.attention as comfy_attention

from .components import ComponentHandle
from .cache_writeback import AsyncCacheWriter
from .aimdo_cache import (
    AimdoCacheController,
    is_aimdo_value,
)
from .qkv import Int8Carrier, carrier_nbytes, install_wan_qkv_patches
from .sparse_backend import SPARSE_BACKEND_OPTION
from .streamed_block import install_streamed_wan_block_patches


# The sampler writes its runtime into per-call transformer options. ComfyUI can
# cache the patched MODEL and SAMPLER nodes independently between executions,
# so callbacks stored on the model must not assume they still share the same
# Python runtime instance as the sampler executing the current prompt.
ACTIVE_RUNTIME_OPTION = "flashvsr_active_runtime"


@dataclass
class PreparedVideo:
    tensor: torch.Tensor  # CPU BCFHW, normalized to [-1, 1]
    latent: dict
    original_frames: int
    generated_frames: int
    crop_start: int
    width: int
    height: int
    output_width: int
    output_height: int


@dataclass(frozen=True)
class CompactKVDescriptor:
    """Chronological compact cache plus the current floating-point K/V."""

    slots: tuple
    current_k: torch.Tensor
    current_v: torch.Tensor
    tokens_per_frame: int
    history_frames: int
    heads: int
    k_reference_mean: torch.Tensor


def prepare_video(images: torch.Tensor) -> PreparedVideo:
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(
            "Expected ComfyUI IMAGE input with shape "
            "[frames, height, width, channels]."
        )
    original_frames, source_h, source_w, _ = images.shape
    if original_frames < 1:
        raise ValueError("At least one input frame is required.")
    if source_h < 1 or source_w < 1:
        raise ValueError("Prepared frames must have a positive size.")

    # The input is already at the desired output resolution. Only perform the
    # FlashVSR-specific right/bottom alignment and normalized CPU packing.
    target_h = math.ceil(source_h / 128) * 128
    target_w = math.ceil(source_w / 128) * 128

    # Match the official FlashVSR temporal layout: source frame zero is the
    # first conditioning frame and all alignment padding is appended at the
    # tail. Pad to the next valid 8n+1 length instead of truncating so ComfyUI
    # can still return every requested source frame.
    required_f = max(25, original_frames + 4)
    padded_f = math.ceil((required_f - 1) / 8) * 8 + 1

    # Allocate the final BCFHW layout directly. The previous frame-major
    # allocation required a second full-video copy after permuting to BCFHW.
    video = torch.empty(
        (1, 3, padded_f, target_h, target_w),
        device="cpu",
        dtype=torch.float16,
    )

    # Bound the FP32 normalization plus FP16 transfer workspace to roughly
    # 128 MiB, reducing Python/CUDA synchronization without exposing another
    # workflow control. On CUDA, also stay within 5% of currently free VRAM.
    # Six bytes per RGB value: one FP32 normalized batch plus its FP16 packed
    # transfer buffer. Keep this arithmetic independent of torch.dtype API
    # differences across supported PyTorch releases.
    bytes_per_frame = 3 * source_h * source_w * 6
    working_budget = 128 * 1024 * 1024
    if images.device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(images.device)
            working_budget = min(
                working_budget,
                max(bytes_per_frame, int(free_bytes * 0.05)),
            )
        except (RuntimeError, TypeError):
            pass
    batch_size = max(
        1,
        min(original_frames, working_budget // max(1, bytes_per_frame)),
    )

    for start in range(0, original_frames, batch_size):
        end = min(start + batch_size, original_frames)
        normalized = (
            images[start:end, ..., :3]
            .movedim(-1, 1)
            .float()
            .mul(2.0)
            .sub(1.0)
        )
        packed = normalized.to(device="cpu", dtype=torch.float16)
        current = video[0, :, start:end]
        current[:, :, :source_h, :source_w].copy_(
            packed.permute(1, 0, 2, 3)
        )
        del normalized, packed

        # Replicate the right and bottom borders directly in the destination,
        # avoiding one padded allocation per source frame.
        if target_w > source_w:
            current[:, :, :source_h, source_w:].copy_(
                current[:, :, :source_h, source_w - 1:source_w].expand(
                    -1, -1, -1, target_w - source_w
                )
            )
        if target_h > source_h:
            current[:, :, source_h:, :].copy_(
                current[:, :, source_h - 1:source_h, :].expand(
                    -1, -1, target_h - source_h, -1
                )
            )

    video[:, :, original_frames:].copy_(
        video[:, :, original_frames - 1:original_frames].expand(
            -1, -1, padded_f - original_frames, -1, -1
        )
    )

    latent_t = (padded_f - 1) // 4
    latent = {
        "samples": torch.zeros(
            (1, 16, latent_t, target_h // 8, target_w // 8),
            dtype=torch.float32,
        )
    }
    return PreparedVideo(
        tensor=video,
        latent=latent,
        original_frames=original_frames,
        generated_frames=padded_f - 4,
        crop_start=0,
        width=target_w,
        height=target_h,
        output_width=source_w,
        output_height=source_h,
    )


class StreamingAttentionDispatcher:
    """FlashVSR overlap-context dispatcher with per-chunk LCSA routing.

    Stock Wan applies RoPE before dispatching to ComfyUI's optimized-attention
    hook. Dispatching at this boundary preserves position-encoded Q/K while
    allowing the selected ComfyUI/KJ backend to perform the calculation.
    """

    def __init__(self, runtime: "FlashVSRRuntime"):
        self.runtime = runtime

    def __call__(self, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False,
                 transformer_options=None, **kwargs):
        options = transformer_options or {}
        runtime = options.get(ACTIVE_RUNTIME_OPTION, self.runtime)
        if not isinstance(runtime, FlashVSRRuntime):
            runtime = self.runtime

        capture_block0 = runtime.profile_should_capture_block0_attention(
            q, k, v, mask, skip_reshape, options
        )
        if capture_block0:
            runtime.profile_activation("q_post_rope", q)
            runtime.profile_activation("k_post_rope", k)

        if not runtime.streaming_active:
            result = runtime.call_attention_backend(
                q, k, v, heads, mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                transformer_options=options,
                **kwargs,
            )
        else:
            result = runtime.streaming_attention(
                q, k, v, heads, mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                transformer_options=options,
                **kwargs,
            )

        if capture_block0:
            # WanSelfAttention applies self_attn.o after this dispatcher.
            runtime.profile_activation("self_attn_pre_o", result)
        return result


class FlashVSRKVCache:
    """Per-Wan-block sliding post-RoPE KV cache.

    Full faithful mode retains three chronological two-frame slots (six
    historical frames). Low-VRAM mode retains the nearest two-frame slot in
    every Wan block. The write cursor always identifies the oldest slot.
    Cached tensors have already received Wan RoPE and must never be rotated
    again.
    """

    PREFILL_FRAMES = 6
    FULL_HISTORY_FRAMES = 6
    LOWVRAM_HISTORY_FRAMES = 2
    SLOT_FRAMES = 2

    def __init__(self, runtime: "FlashVSRRuntime"):
        self.runtime = runtime
        self.mode = None
        self.total_blocks = 0
        self.active_blocks = set()
        self.history_frames = 0
        self.slot_count = 0
        self.entries = {}
        self.pending_entries = {}
        self.cache_format = "int8"
        self.residency_backend = "cpu"
        self.aimdo_controller = None
        self.bytes_per_block = 0
        self.write_slot = 0
        self.initial_chunk = False
        self.committed_blocks = set()
        self.stage_k = None
        self.stage_v = None
        self.total_cache_bytes = 0
        self.reported = False
        self.k_reference_means = {}
        self.async_writer = None

    @property
    def enabled(self):
        return bool(self.active_blocks)

    def configure(self, mode, total_blocks, cache_format="int8",
                  residency_backend="cpu"):
        self.clear()
        self.mode = mode
        self.cache_format = str(cache_format)
        self.residency_backend = str(residency_backend)
        if self.cache_format not in ("int8", "hybrid", "float"):
            raise ValueError(
                f"Unknown FlashVSR faithful cache format: {self.cache_format}"
            )
        if self.residency_backend not in ("cpu", "aimdo_experimental"):
            raise ValueError(
                "Unknown FlashVSR cache residency backend: "
                f"{self.residency_backend}"
            )
        self.total_blocks = max(0, int(total_blocks or 0))
        if mode == "streaming_faithful_full":
            self.active_blocks = set(range(self.total_blocks))
            self.history_frames = self.FULL_HISTORY_FRAMES
        elif mode == "streaming_faithful_lowvram":
            # Immediate history in every layer is substantially more useful
            # than a longer history introduced only after early blocks have
            # already processed the continuation without temporal context.
            # For a 30-block Wan model this has the same block-frame cache
            # footprint as retaining six frames in only the final ten blocks.
            self.active_blocks = set(range(self.total_blocks))
            self.history_frames = self.LOWVRAM_HISTORY_FRAMES
        else:
            self.active_blocks = set()
            self.history_frames = 0
        self.slot_count = self.history_frames // self.SLOT_FRAMES
        if mode and not self.active_blocks:
            raise RuntimeError(
                "FlashVSR could not determine the Wan transformer block "
                "count required for faithful KV caching."
            )

    def participates(self, block_index):
        return int(block_index) in self.active_blocks

    def begin_chunk(self, initial):
        self.initial_chunk = bool(initial)
        self.committed_blocks.clear()
        self.pending_entries = {}

    def end_chunk(self):
        if not self.enabled:
            return
        missing = self.active_blocks.difference(self.committed_blocks)
        if missing:
            preview = ", ".join(str(index) for index in sorted(missing)[:8])
            raise RuntimeError(
                "FlashVSR did not update every configured KV-cache block "
                f"during this model call. Missing block indices: {preview}."
            )
        # CPU cache placeholders returned by the bounded asynchronous writer
        # must be complete before they become authoritative. Most transfers
        # have already overlapped later Wan blocks by this point.
        if self.async_writer is not None:
            self.async_writer.flush()
        if self.initial_chunk:
            self.entries = self.pending_entries
        else:
            for block_index, pending in self.pending_entries.items():
                committed = []
                for value in pending:
                    if (
                        isinstance(value, tuple)
                        and len(value) == 2
                        and is_aimdo_value(value[0])
                    ):
                        wrapper, update = value
                        wrapper.commit_update(update)
                        committed.append(wrapper)
                    else:
                        committed.append(value)
                self.entries[block_index][self.write_slot] = tuple(committed)
        if not self.initial_chunk:
            self.write_slot = (self.write_slot + 1) % self.slot_count
        self.pending_entries = {}

    @staticmethod
    def _allocate_copy(source, device):
        target = torch.device(device)
        destination = torch.empty(
            source.shape, device=target, dtype=source.dtype
        )
        destination.copy_(source.detach(), non_blocking=False)
        return destination

    @staticmethod
    def _carrier_to(carrier, device):
        if not isinstance(carrier, Int8Carrier):
            return FlashVSRKVCache._allocate_copy(carrier, device)
        target = torch.device(device)
        return Int8Carrier(
            carrier.qdata.to(device=target).contiguous(),
            carrier.scale.to(device=target).contiguous(),
            carrier.shape,
            carrier.dtype,
            carrier.head_dim,
        )

    def _make_gpu_value(self, source, value_kind, heads):
        compact = (
            self.cache_format == "int8"
            or (self.cache_format == "hybrid" and value_kind == "v")
        )
        if compact:
            return Int8Carrier.from_tensor(
                source, source.device, heads
            )
        return source.detach()

    def _make_cache_value(self, source, value_kind, heads):
        gpu_value = self._make_gpu_value(source, value_kind, heads)
        return self._carrier_to(
            gpu_value, torch.device("cpu")
        ), gpu_value

    def _store_new_value(self, source, value_kind, heads):
        cpu_value, gpu_value = self._make_cache_value(
            source, value_kind, heads
        )
        if self.aimdo_controller is not None:
            return self.aimdo_controller.wrap(cpu_value, gpu_value)
        return cpu_value

    def _make_k_summary(self, source, heads, tokens_per_frame):
        """Pool live K while its FP tensor is already available at commit.

        v0.34 reconstructed summaries from the newly quantized carrier. That
        reread INT8 data and row scales immediately after quantization. The
        summary is routing metadata, so compute it directly from the same live
        K projection before the compact CPU write-through begins.
        """
        if not torch.is_tensor(source) or source.ndim != 3:
            raise TypeError("Compact LCSA summaries require a BNC K tensor.")
        batch, tokens, channels = source.shape
        head_dim = channels // heads
        height = self.runtime.video.height // 16
        width = self.runtime.video.width // 16
        if tokens != self.SLOT_FRAMES * tokens_per_frame:
            raise RuntimeError("Unexpected compact K slot length.")
        summary_marker = self.runtime.profile_start(source)
        block_indices, _ = self.runtime._lcsa_token_indices(
            self.SLOT_FRAMES, height, width, source.device
        )
        block_count = tokens // 128
        pooled = torch.empty(
            (batch, heads, block_count, head_dim),
            device=source.device,
            dtype=source.dtype,
        )
        values_hnd = source.view(batch, tokens, heads, head_dim).permute(
            0, 2, 1, 3
        )
        # Keep index-select work bounded at high resolutions.
        blocks_per_chunk = 128
        for start in range(0, block_count, blocks_per_chunk):
            end = min(start + blocks_per_chunk, block_count)
            indices = block_indices[start * 128:end * 128]
            pooled[:, :, start:end].copy_(
                values_hnd.index_select(2, indices)
                .view(batch, heads, end - start, 128, head_dim)
                .mean(3)
            )
        # One slot is one logical temporal block. Store the tiny summary on
        # CPU; unlike K itself it is copied once per layer, not per token.
        summary = pooled.permute(0, 2, 1, 3).unsqueeze(1).to(
            device="cpu", non_blocking=False
        ).contiguous()
        self.runtime.profile_end("kv_k_summary_build_d2h", summary_marker)
        return summary

    def _store_new_slot(self, k, v, heads, tokens_per_frame):
        gpu_k = self._make_gpu_value(k, "k", heads)
        gpu_v = self._make_gpu_value(v, "v", heads)
        if self.async_writer is not None and self.cache_format == "int8":
            cpu_k, cpu_v = self.async_writer.submit_pair(gpu_k, gpu_v)
        else:
            cpu_k = self._carrier_to(gpu_k, torch.device("cpu"))
            cpu_v = self._carrier_to(gpu_v, torch.device("cpu"))
        summary = (
            self._make_k_summary(k, heads, tokens_per_frame)
            if self.cache_format == "int8" else None
        )
        if self.aimdo_controller is not None:
            cpu_k = self.aimdo_controller.wrap(cpu_k, gpu_k)
            cpu_v = self.aimdo_controller.wrap(cpu_v, gpu_v)
        return cpu_k, cpu_v, summary

    @staticmethod
    def _copy_cached_to(cached, destination):
        if is_aimdo_value(cached):
            cached.copy_to(destination)
        elif isinstance(cached, Int8Carrier):
            cached.copy_to(destination)
        else:
            destination.copy_(cached, non_blocking=False)

    def _slot_bytes(self, shape, heads, element_size):
        float_bytes = math.prod(shape) * element_size
        int8_bytes = carrier_nbytes(shape, heads)
        if self.cache_format == "int8":
            return int8_bytes * 2
        if self.cache_format == "hybrid":
            return float_bytes + int8_bytes
        return float_bytes * 2

    def _initialize_storage(self, k, slot_tokens, heads):
        slot_shape = (k.shape[0], slot_tokens, k.shape[2])
        bytes_per_slot = self._slot_bytes(
            slot_shape, heads, k.element_size()
        )
        self.bytes_per_block = self.slot_count * bytes_per_slot
        self.total_cache_bytes = (
            self.bytes_per_block * len(self.active_blocks)
        )
        if self.cache_format == "int8" and k.device.type == "cuda":
            self.async_writer = AsyncCacheWriter(
                self.runtime, k.device, depth=2
            )
            if not self.async_writer.enabled:
                self.async_writer.report()
        if self.residency_backend == "aimdo_experimental":
            compact_components = 2
            k_components = (
                compact_components if self.cache_format == "int8" else 1
            )
            v_components = (
                compact_components
                if self.cache_format in ("int8", "hybrid") else 1
            )
            allocation_count = (
                len(self.active_blocks) * self.slot_count
                * (k_components + v_components)
            )
            controller = AimdoCacheController(
                self.total_cache_bytes, allocation_count, k.device,
                runtime=self.runtime,
            )
            if controller.enabled:
                self.aimdo_controller = controller
            else:
                controller.report()
                self.aimdo_controller = None
        if not self.reported:
            first = min(self.active_blocks)
            last = max(self.active_blocks)
            actual = (
                "aimdo_experimental"
                if self.aimdo_controller is not None else "cpu"
            )
            print(
                "[FlashVSR] faithful KV cache: blocks "
                f"{first}-{last}, history={self.history_frames} frames, "
                f"format={self.cache_format}, residency={actual}, "
                f"CPU-authoritative={self.total_cache_bytes / (1024 ** 3):.2f} GiB."
            )
            print(
                "[FlashVSR] faithful KV cache maximum CPU fallback per "
                "continuation: approximately "
                f"{self.total_cache_bytes / (1024 ** 3):.2f} GiB "
                "CPU->GPU and "
                f"{self.total_cache_bytes / self.slot_count / (1024 ** 3):.2f} GiB "
                "GPU->CPU. AIMDO resident hits reduce CPU->GPU traffic."
            )
            self.reported = True

    def _copy_initial_slots(self, block_index, k, v, slot_tokens, heads,
                            tokens_per_frame):
        slots = []
        # Full mode retains all six prefill frames. Low-VRAM mode must retain
        # the nearest two prefill frames, so start at the tail of the input.
        history_tokens = self.history_frames * (slot_tokens // self.SLOT_FRAMES)
        history_start = k.shape[1] - history_tokens
        for slot_index in range(self.slot_count):
            start = history_start + slot_index * slot_tokens
            end = start + slot_tokens
            slots.append(self._store_new_slot(
                k[:, start:end], v[:, start:end], heads, tokens_per_frame
            ))
        return slots

    def describe(self, block_index, k, v, tokens_per_frame, heads):
        """Return compact chronological cache metadata without FP expansion."""
        if self.initial_chunk or self.cache_format != "int8":
            return None
        slots = self.entries.get(int(block_index))
        if slots is None:
            raise RuntimeError(
                f"FlashVSR KV cache for Wan block {block_index} was not "
                "initialized by the first six-frame model call."
            )
        chronological = tuple(
            slots[(self.write_slot + relative) % self.slot_count]
            for relative in range(self.slot_count)
        )
        if any(len(slot) != 3 or slot[2] is None for slot in chronological):
            return None
        return CompactKVDescriptor(
            chronological, k, v, tokens_per_frame,
            self.history_frames, heads,
            self.k_reference_means[int(block_index)],
        )

    @contextmanager
    def acquire_compact_slots(self, descriptor, device):
        """Pin/acquire every compact slot for one native Sparge launch."""
        with ExitStack() as stack:
            acquired = []
            for cached_k, cached_v, _summary in descriptor.slots:
                compact_k = stack.enter_context(
                    self._acquire_compact(cached_k, device)
                )
                compact_v = stack.enter_context(
                    self._acquire_compact(cached_v, device)
                )
                acquired.append((compact_k, compact_v))
            yield tuple(acquired)

    def stage(self, block_index, k, v, tokens_per_frame):
        """Return chronological cached+current K/V for a continuation."""
        if self.initial_chunk:
            return k, v
        slots = self.entries.get(int(block_index))
        if slots is None:
            raise RuntimeError(
                f"FlashVSR KV cache for Wan block {block_index} was not "
                "initialized by the first six-frame model call."
            )
        current_tokens = self.SLOT_FRAMES * tokens_per_frame
        history_tokens = self.history_frames * tokens_per_frame
        total_tokens = history_tokens + current_tokens
        required_shape = (k.shape[0], total_tokens, k.shape[2])
        if (
            self.stage_k is None
            or tuple(self.stage_k.shape) != required_shape
            or self.stage_k.device != k.device
            or self.stage_k.dtype != k.dtype
        ):
            self.stage_k = torch.empty(
                required_shape, device=k.device, dtype=k.dtype
            )
            self.stage_v = torch.empty_like(self.stage_k)

        offset = 0
        for relative in range(self.slot_count):
            slot_index = (self.write_slot + relative) % self.slot_count
            cached_k, cached_v = slots[slot_index][:2]
            end = offset + current_tokens
            self._copy_cached_to(
                cached_k, self.stage_k[:, offset:end]
            )
            self._copy_cached_to(
                cached_v, self.stage_v[:, offset:end]
            )
            offset = end
        self.stage_k[:, history_tokens:].copy_(k, non_blocking=False)
        self.stage_v[:, history_tokens:].copy_(v, non_blocking=False)
        return self.stage_k, self.stage_v

    @contextmanager
    def _acquire_compact(self, cached, device):
        if is_aimdo_value(cached):
            with cached.acquire_compact(device) as value:
                yield value
            return
        if not isinstance(cached, Int8Carrier):
            raise TypeError(
                "Direct FlashVSR HND materialization requires an INT8 cache."
            )
        value = self._carrier_to(cached, device)
        try:
            yield value
        finally:
            del value

    @staticmethod
    def _dequantize_hnd(carrier, destination, block_indices):
        if not isinstance(carrier, Int8Carrier):
            raise TypeError("Expected a compact INT8 cache carrier.")
        batch, tokens, channels = carrier.shape
        heads = destination.shape[1]
        head_dim = channels // heads
        if destination.shape != (batch, heads, tokens, head_dim):
            raise RuntimeError("Compact cache HND destination shape mismatch.")
        qdata = carrier.qdata.view(batch, tokens, heads, head_dim).permute(
            0, 2, 1, 3
        )
        scale = carrier.scale.view(batch, tokens, heads, 1).permute(
            0, 2, 1, 3
        )
        # Bound index-select work while writing directly into the final HND
        # allocation consumed by SpargeAttn.
        tokens_per_chunk = 4096
        for start in range(0, tokens, tokens_per_chunk):
            end = min(start + tokens_per_chunk, tokens)
            indices = block_indices[start:end]
            target = destination[:, :, start:end]
            target.copy_(qdata.index_select(2, indices))
            target.mul_(
                scale.index_select(2, indices).to(dtype=destination.dtype)
            )

    def materialize_hnd(self, descriptor, block_indices, profiler=None):
        """Expand compact slots once, directly into final Sparge HND layout."""
        current_k = descriptor.current_k
        current_v = descriptor.current_v
        batch, current_tokens, channels = current_k.shape
        heads = descriptor.heads
        head_dim = channels // heads
        slot_tokens = self.SLOT_FRAMES * descriptor.tokens_per_frame
        history_tokens = descriptor.history_frames * descriptor.tokens_per_frame
        total_tokens = history_tokens + current_tokens
        k_hnd = torch.empty(
            (batch, heads, total_tokens, head_dim),
            device=current_k.device,
            dtype=current_k.dtype,
        )
        v_hnd = torch.empty_like(k_hnd)

        offset = 0
        for cached_k, cached_v, _summary in descriptor.slots:
            end = offset + slot_tokens
            for cached, destination in (
                (cached_k, k_hnd[:, :, offset:end]),
                (cached_v, v_hnd[:, :, offset:end]),
            ):
                transfer_marker = (
                    profiler.profile_start(current_k)
                    if profiler is not None else None
                )
                with self._acquire_compact(cached, current_k.device) as carrier:
                    if profiler is not None:
                        profiler.profile_end(
                            "kv_cache_compact_h2d", transfer_marker
                        )
                    dequant_marker = (
                        profiler.profile_start(current_k)
                        if profiler is not None else None
                    )
                    self._dequantize_hnd(
                        carrier, destination, block_indices
                    )
                    if profiler is not None:
                        profiler.profile_end(
                            "kv_cache_dequant_hnd", dequant_marker
                        )
            offset = end

        current_marker = (
            profiler.profile_start(current_k)
            if profiler is not None else None
        )
        for source, destination in (
            (current_k, k_hnd[:, :, history_tokens:]),
            (current_v, v_hnd[:, :, history_tokens:]),
        ):
            destination.copy_(
                source.view(batch, current_tokens, heads, head_dim)
                .permute(0, 2, 1, 3)
                .index_select(2, block_indices)
            )
        if profiler is not None:
            profiler.profile_end("kv_current_hnd", current_marker)
        return k_hnd, v_hnd

    def materialize_v_hnd(self, descriptor, block_indices, profiler=None):
        """Materialize only FP16/BF16 V for Sparge's Ampere ABI."""
        current = descriptor.current_v
        batch, current_tokens, channels = current.shape
        heads = descriptor.heads
        head_dim = channels // heads
        slot_tokens = self.SLOT_FRAMES * descriptor.tokens_per_frame
        history_tokens = descriptor.history_frames * descriptor.tokens_per_frame
        total_tokens = history_tokens + current_tokens
        output = torch.empty(
            (batch, heads, total_tokens, head_dim),
            device=current.device, dtype=current.dtype,
        )
        offset = 0
        for _cached_k, cached_v, _summary in descriptor.slots:
            end = offset + slot_tokens
            transfer_marker = (
                profiler.profile_start(current) if profiler else None
            )
            with self._acquire_compact(cached_v, current.device) as carrier:
                if profiler:
                    profiler.profile_end(
                        "kv_cache_compact_h2d", transfer_marker
                    )
                marker = profiler.profile_start(current) if profiler else None
                self._dequantize_hnd(
                    carrier, output[:, :, offset:end], block_indices
                )
                if profiler:
                    profiler.profile_end("kv_cache_dequant_hnd", marker)
            offset = end
        marker = profiler.profile_start(current) if profiler else None
        output[:, :, history_tokens:].copy_(
            current.view(batch, current_tokens, heads, head_dim)
            .permute(0, 2, 1, 3)
            .index_select(2, block_indices)
        )
        if profiler:
            profiler.profile_end("kv_current_hnd", marker)
        return output

    def commit(self, block_index, k, v, tokens_per_frame, heads):
        block_index = int(block_index)
        if self.initial_chunk:
            expected = self.PREFILL_FRAMES * tokens_per_frame
            if k.shape[1] != expected or v.shape[1] != expected:
                raise RuntimeError(
                    "FlashVSR initial KV cache requires exactly six latent "
                    "frames."
                )
            slot_tokens = self.SLOT_FRAMES * tokens_per_frame
            if self.total_cache_bytes == 0:
                self._initialize_storage(k, slot_tokens, heads)
            batch, _tokens, channels = k.shape
            head_dim = channels // heads
            self.k_reference_means[block_index] = (
                k.detach()
                .view(batch, -1, heads, head_dim)
                .float()
                .mean(dim=1)
                .to(device="cpu", non_blocking=False)
                .contiguous()
            )
            pending = self._copy_initial_slots(
                block_index, k, v, slot_tokens, heads, tokens_per_frame
            )
            self.pending_entries[block_index] = pending
        else:
            expected = self.SLOT_FRAMES * tokens_per_frame
            if k.shape[1] != expected or v.shape[1] != expected:
                raise RuntimeError(
                    "FlashVSR continuation KV cache requires exactly two "
                    "new latent frames."
                )
            cached_k, cached_v, _cached_summary = (
                self.entries[block_index][self.write_slot]
            )
            gpu_k = self._make_gpu_value(k, "k", heads)
            gpu_v = self._make_gpu_value(v, "v", heads)
            if self.async_writer is not None and self.cache_format == "int8":
                cpu_k, cpu_v = self.async_writer.submit_pair(gpu_k, gpu_v)
            else:
                cpu_k = self._carrier_to(gpu_k, torch.device("cpu"))
                cpu_v = self._carrier_to(gpu_v, torch.device("cpu"))
            summary = (
                self._make_k_summary(k, heads, tokens_per_frame)
                if self.cache_format == "int8" else None
            )
            pending = []
            for cached, cpu_value, gpu_source in (
                (cached_k, cpu_k, gpu_k),
                (cached_v, cpu_v, gpu_v),
            ):
                if is_aimdo_value(cached):
                    pending.append((
                        cached,
                        cached.prepare_update(cpu_value, gpu_source),
                    ))
                else:
                    pending.append(cpu_value)
            pending.append(summary)
            self.pending_entries[block_index] = tuple(pending)
        self.committed_blocks.add(block_index)

    def clear(self):
        writer = self.async_writer
        if writer is not None:
            try:
                writer.close()
                writer.report()
            except Exception as error:
                print(
                    "[FlashVSR] async cache writer cleanup failed: "
                    f"{error}."
                )
        controller = self.aimdo_controller
        if controller is not None:
            try:
                if controller.device.type == "cuda":
                    torch.cuda.synchronize(controller.device)
            except Exception:
                pass
            controller.report()
        self.mode = None
        self.total_blocks = 0
        self.entries = {}
        self.pending_entries = {}
        self.active_blocks = set()
        self.history_frames = 0
        self.slot_count = 0
        self.aimdo_controller = None
        self.bytes_per_block = 0
        self.write_slot = 0
        self.initial_chunk = False
        self.committed_blocks = set()
        self.stage_k = None
        self.stage_v = None
        self.total_cache_bytes = 0
        self.reported = False
        self.k_reference_means = {}
        self.async_writer = None


class FlashVSRRuntime:
    """FlashVSR LQ conditioning and optional temporal streaming state."""

    def __init__(self, lq: ComponentHandle, video: PreparedVideo,
                 conditioning_strength: float):
        self.lq = lq
        self.video = video
        self.conditioning_strength = conditioning_strength
        self.current_lq = None
        self.lq_overlap_tail = None
        self.lq_overlap_compact = False
        self.lq_segment_buffers = None
        self.current_latent_frames = 0
        self.current_rope_start = 0
        self.streaming_active = False
        self.sampling_mode = None
        self.total_wan_blocks = 0
        self.lcsa_sparse_ratio = 2.0
        self.lcsa_local_range = 11
        self.lcsa_query_block_chunk = 1
        self.resolved_lcsa_query_block_chunk = {}
        self.local_spatial_mask_cache = {}
        self.local_topology_cache = {}
        self.lcsa_token_index_cache = {}
        self.reported_auto_chunks = set()
        self.seen_self_attention_blocks = set()
        self.attention_backend = None
        self.sparse_attention_backend = None
        self.attention_override = StreamingAttentionDispatcher(self)
        self.installed_attention_override = None
        self.dynamic_vram_active = False
        self.qkv_projection_mode = "stock"
        self.int8_qkv_capable = False
        self.int8_qkv_format = None
        self.int8_qkv_run_active = False
        self.int8_qkv_disabled_reason = None
        self.reported_qkv_path = None
        self.profile_enabled = False
        self.profile_events = {}
        self.profile_devices = set()
        self.profile_counters = {}
        self.profile_activation_stats = {}
        self.kv_cache = FlashVSRKVCache(self)

    def begin_profile(self, enabled: bool):
        """Start a non-synchronizing CUDA-event profiling collection."""
        self.profile_enabled = bool(enabled and torch.cuda.is_available())
        self.profile_events = {}
        self.profile_devices = set()
        self.profile_counters = {}
        self.profile_activation_stats = {}
        if enabled and not self.profile_enabled:
            print(
                "[FlashVSR profiler] CUDA is unavailable; profiling disabled."
            )

    def profile_start(self, tensor_or_device):
        source_device = getattr(
            tensor_or_device, "device", tensor_or_device
        )
        device = torch.device(source_device)
        if not self.profile_enabled or device.type != "cuda":
            return None
        with torch.cuda.device(device):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        self.profile_devices.add(device)
        return event, device

    def profile_end(self, name: str, marker):
        if marker is None:
            return
        start, device = marker
        with torch.cuda.device(device):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
        self.profile_events.setdefault(name, []).append(
            (start, end, device)
        )

    def profile_record(self, name, start, end, device):
        """Register events recorded on a non-default CUDA stream."""
        if not self.profile_enabled:
            return
        device = torch.device(device)
        self.profile_devices.add(device)
        self.profile_events.setdefault(name, []).append(
            (start, end, device)
        )

    def profile_count(self, name, value, replace=False):
        if not self.profile_enabled:
            return
        if replace:
            self.profile_counters[name] = value
        else:
            self.profile_counters[name] = (
                self.profile_counters.get(name, 0) + value
            )

    def _profile_first_prefill_active(self):
        return (
            self.profile_enabled
            and self.current_rope_start == 0
            and self.current_latent_frames == 6
        )

    def profile_should_capture_block0_attention(
        self, q, k, v, mask, skip_reshape, transformer_options
    ):
        """Identify block-0 self-attention in the six-frame prefill."""
        if not self._profile_first_prefill_active():
            return False
        if mask is not None or skip_reshape:
            return False
        if int(transformer_options.get("block_index", -1)) != 0:
            return False
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            return False
        spatial_tokens = (
            (self.video.height // 16) * (self.video.width // 16)
        )
        expected_tokens = self.current_latent_frames * spatial_tokens
        return (
            q.shape[1] == expected_tokens
            and k.shape[1] == expected_tokens
            and v.shape[1] == expected_tokens
        )

    def profile_activation(self, name, tensor):
        """Queue compact temporal activation statistics without CPU sync."""
        if (
            not self._profile_first_prefill_active()
            or name in self.profile_activation_stats
            or not torch.is_tensor(tensor)
            or tensor.ndim != 3
            or tensor.device.type != "cuda"
        ):
            return

        frames = self.current_latent_frames
        if tensor.shape[1] % frames:
            return

        with torch.no_grad():
            values = tensor.detach().reshape(
                tensor.shape[0], frames, -1
            )
            target_values_per_frame = 32768
            stride = max(
                1,
                math.ceil(values.shape[-1] / target_values_per_frame),
            )
            sample = values[..., ::stride].float()

            mean = sample.mean()
            rms = sample.square().mean().sqrt()
            std = (sample - mean).square().mean().sqrt()
            max_abs = sample.abs().amax()

            previous = sample[:, :-1]
            current = sample[:, 1:]
            dot = (previous * current).sum(dim=-1)
            norm = (
                previous.square().sum(dim=-1).sqrt()
                * current.square().sum(dim=-1).sqrt()
            ).clamp_min(1e-12)
            temporal_cos = (dot / norm).mean()
            previous_abs = previous.abs().mean(dim=-1)
            current_abs = current.abs().mean(dim=-1)
            denominator = (
                0.5 * (previous_abs + current_abs)
            ).clamp_min(1e-12)
            temporal_rel_l1 = (
                (previous - current).abs().mean(dim=-1)
                / denominator
            ).mean()

        self.profile_devices.add(tensor.device)
        self.profile_activation_stats[name] = {
            "std": std,
            "rms": rms,
            "max_abs": max_abs,
            "temporal_cos": temporal_cos,
            "temporal_rel_l1": temporal_rel_l1,
        }

    def finish_profile(self):
        """Synchronize once and print aggregate stage timings."""
        if not self.profile_enabled:
            return
        for device in self.profile_devices:
            torch.cuda.synchronize(device)

        totals = {}
        calls = {}
        for name, records in self.profile_events.items():
            totals[name] = sum(
                start.elapsed_time(end) for start, end, _ in records
            )
            calls[name] = len(records)

        print("[FlashVSR profiler] CUDA stage totals:")
        preferred = (
            "sampling_total",
            "lq_projector",
            "lq_input_transfer",
            "lq_input_cast",
            "lq_pixel_unshuffle",
            "lq_conv1",
            "lq_norm_act1",
            "lq_conv2",
            "lq_norm_act2",
            "lq_cache_update",
            "lq_linear",
            "model_total",
            "kv_cache_stage_h2d",
            "kv_cache_compact_h2d",
            "kv_cache_dequant_hnd",
            "kv_current_hnd",
            "kv_k_summary_build_d2h",
            "kv_summary_h2d",
            "kv_cache_commit_enqueue",
            "kv_write_d2h",
            "kv_write_wait_for_slot",
            "kv_cached_attention",
            "qkv_convrot_quant",
            "qkv_q_gemm",
            "qkv_k_gemm",
            "qkv_v_gemm",
            "qkv_q_norm_rope",
            "qkv_k_norm_rope",
            "lcsa_routing",
            "sparge_layout_qkv",
            "sparge_mask_convert",
            "sparge_kernel",
            "sparge_input_cast",
            "sparge_k_smooth",
            "sparge_q_quant",
            "sparge_k_scale_reduce",
            "sparge_k_int8_to_block_int8",
            "sparge_k_native_total",
            "sparge_qk_quant",
            "sparge_lut",
            "sparge_v_transpose",
            "sparge_v_quant",
            "sparge_v_scale_reduce",
            "sparge_v_int8_to_fp8",
            "sparge_v_current_to_fp8",
            "sparge_v_native_total",
            "sparge_attention_cuda",
            "sparge_restore",
            "result_assembly",
        )
        for name in preferred:
            if name not in totals:
                continue
            total = totals[name]
            count = calls[name]
            print(
                f"[FlashVSR profiler]   {name}: {total:.2f} ms "
                f"({count} calls, {total / count:.2f} ms/call)"
            )
        if self.profile_activation_stats:
            print(
                "[FlashVSR profiler] block0 first-prefill activation "
                "diagnostics (sampled; timings include diagnostic kernels):"
            )
            print(
                "[FlashVSR profiler]   stage                    "
                "rms       std       max_abs   temp_cos  temp_rel_l1"
            )
            for name in (
                "patch_tokens",
                "lq_tokens",
                "patch_plus_lq",
                "q_post_rope",
                "k_post_rope",
                "self_attn_pre_o",
                "block0_output",
            ):
                stats = self.profile_activation_stats.get(name)
                if stats is None:
                    continue
                print(
                    f"[FlashVSR profiler]   {name:<24} "
                    f"{stats['rms'].item():8.4f} "
                    f"{stats['std'].item():9.4f} "
                    f"{stats['max_abs'].item():9.4f} "
                    f"{stats['temporal_cos'].item():9.4f} "
                    f"{stats['temporal_rel_l1'].item():12.4f}"
                )
            patch_stats = self.profile_activation_stats.get("patch_tokens")
            lq_stats = self.profile_activation_stats.get("lq_tokens")
            if patch_stats is not None and lq_stats is not None:
                ratio = (
                    lq_stats["rms"].item()
                    / max(patch_stats["rms"].item(), 1e-12)
                )
                print(
                    f"[FlashVSR profiler]   LQ / patch RMS: {ratio:.4f}"
                )
        for name in (
            "kv_write_enqueue", "kv_write_bytes",
            "aimdo_fault_success", "aimdo_fault_failure",
        ):
            if name in self.profile_counters:
                value = self.profile_counters[name]
                suffix = (
                    f" ({value / (1024 ** 3):.2f} GiB)"
                    if name.endswith("bytes") else ""
                )
                print(f"[FlashVSR profiler]   {name}: {value}{suffix}")

        nested = sum(
            totals.get(name, 0.0) for name in (
                "kv_cache_stage_h2d",
                "kv_cache_compact_h2d",
                "kv_cache_dequant_hnd",
                "kv_current_hnd",
                "kv_cache_commit_enqueue",
                "qkv_convrot_quant",
                "qkv_q_gemm",
                "qkv_k_gemm",
                "qkv_v_gemm",
                "qkv_q_norm_rope",
                "qkv_k_norm_rope",
                "lcsa_routing",
                "sparge_layout_qkv",
                "sparge_mask_convert",
                "sparge_kernel",
                "sparge_restore",
            )
        )
        if "model_total" in totals:
            unattributed = max(0.0, totals["model_total"] - nested)
            print(
                "[FlashVSR profiler]   model_unattributed: "
                f"{unattributed:.2f} ms (Wan projections, MLP, cross-"
                "attention, normalization, and unprofiled backend work)"
            )
        sparge_detail = sum(
            totals.get(name, 0.0) for name in (
                "sparge_input_cast",
                "sparge_k_smooth",
                "sparge_qk_quant",
                "sparge_q_quant",
                "sparge_k_native_total",
                "sparge_lut",
                "sparge_v_transpose",
                "sparge_v_quant",
                "sparge_v_scale_reduce",
                "sparge_v_native_total",
                "sparge_attention_cuda",
            )
        )
        if "sparge_kernel" in totals and sparge_detail:
            unattributed = max(0.0, totals["sparge_kernel"] - sparge_detail)
            print(
                "[FlashVSR profiler]   sparge_internal_unattributed: "
                f"{unattributed:.2f} ms (allocation, dispatch, and "
                "SpargeAttn wrapper overhead)"
            )
        lq_nested = sum(
            totals.get(name, 0.0) for name in (
                "lq_input_transfer",
                "lq_input_cast",
                "lq_pixel_unshuffle",
                "lq_conv1",
                "lq_norm_act1",
                "lq_conv2",
                "lq_norm_act2",
                "lq_cache_update",
                "lq_linear",
            )
        )
        if "lq_projector" in totals:
            lq_unattributed = max(
                0.0, totals["lq_projector"] - lq_nested
            )
            print(
                "[FlashVSR profiler]   lq_unattributed: "
                f"{lq_unattributed:.2f} ms (Python dispatch, output-buffer "
                "copies, and unprofiled projector work)"
            )
        self.profile_enabled = False
        self.profile_events = {}
        self.profile_devices = set()
        self.profile_counters = {}
        self.profile_activation_stats = {}

    def set_attention_backend(self, backend):
        self.attention_backend = self._unwrap_attention_backend(backend)

    @staticmethod
    def _unwrap_attention_backend(backend):
        """Return the real backend beneath any FlashVSR dispatcher wrappers.

        ``ModelPatcher.set_model_optimized_attention`` wraps the installed
        callable in a closure. We tag that closure with ``flashvsr_runtime``;
        on a warm run it must be unwrapped rather than captured as the next
        runtime's fallback backend, which would recurse back into FlashVSR.
        """
        seen = set()
        while backend is not None and id(backend) not in seen:
            seen.add(id(backend))
            if isinstance(backend, StreamingAttentionDispatcher):
                owner = backend.runtime
            else:
                owner = getattr(backend, "flashvsr_runtime", None)
            if not isinstance(owner, FlashVSRRuntime):
                return backend
            backend = owner.attention_backend
        return None

    def set_sparse_attention_backend(self, backend):
        self.sparse_attention_backend = backend

    def set_installed_attention_override(self, override):
        self.installed_attention_override = override
        if override is not None:
            override.flashvsr_runtime = self

    def force_streaming_attention_override(self, transformer_options):
        """Install the streaming dispatcher for this call and preserve backends.

        This also handles a ModelAttentionBackend/KJ node placed after the
        FlashVSR model patch: its override becomes the dispatcher's backend.
        """
        existing = transformer_options.get("optimized_attention_override")
        backend = self._unwrap_attention_backend(existing)
        if backend is not None:
            self.attention_backend = backend
        sparse_backend = transformer_options.get(SPARSE_BACKEND_OPTION)
        if sparse_backend is not None:
            self.set_sparse_attention_backend(sparse_backend)
        if self.installed_attention_override is not None:
            transformer_options["optimized_attention_override"] = (
                self.installed_attention_override
            )

    def call_attention_backend(self, q, k, v, heads, mask=None,
                               attn_precision=None, skip_reshape=False,
                               skip_output_reshape=False,
                               transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        options.pop("optimized_attention_override", None)
        options.pop(SPARSE_BACKEND_OPTION, None)
        call_kwargs = dict(kwargs)
        call_kwargs.pop("_inside_attn_wrapper", None)
        call_kwargs.update({
            "mask": mask,
            "attn_precision": attn_precision,
            "skip_reshape": skip_reshape,
            "skip_output_reshape": skip_output_reshape,
            "transformer_options": options,
        })
        backend = self._unwrap_attention_backend(self.attention_backend)
        # Store the sanitized result as well, repairing a runtime retained
        # after a failed warm execution without requiring another capture.
        self.attention_backend = backend
        if backend is not None:
            return backend(
                comfy_attention.optimized_attention,
                q, k, v, heads,
                **call_kwargs,
            )
        return comfy_attention.optimized_attention(
            q, k, v, heads,
            **call_kwargs,
        )

    def set_total_wan_blocks(self, count):
        self.total_wan_blocks = max(0, int(count))

    def set_dynamic_vram(self, enabled):
        self.dynamic_vram_active = bool(enabled)

    def configure_int8_qkv(self, format_info):
        self.int8_qkv_format = format_info
        self.int8_qkv_capable = format_info is not None
        self.int8_qkv_disabled_reason = None
        if format_info is None:
            print(
                "[FlashVSR] shared QKV: stock projection fallback "
                "(the complete Wan self-attention stack is not compatible "
                "TensorWise INT8 ConvRot)."
            )

    def note_qkv_path(self, path):
        self.int8_qkv_run_active = True
        if path == self.reported_qkv_path:
            return
        self.reported_qkv_path = path
        dynamic = "on" if self.dynamic_vram_active else "off"
        print(
            f"[FlashVSR] shared QKV path: {path}; "
            f"ComfyUI Dynamic VRAM={dynamic}."
        )

    def disable_int8_qkv(self, reason):
        self.int8_qkv_capable = False
        self.int8_qkv_run_active = False
        self.int8_qkv_disabled_reason = str(reason)
        if self.reported_qkv_path != "stock_fallback":
            print(
                "[FlashVSR] shared QKV fell back to stock Wan: "
                f"{self.int8_qkv_disabled_reason}."
            )
            self.reported_qkv_path = "stock_fallback"

    def begin_sampling(self, streaming: bool, sampling_mode=None,
                       sparse_ratio: float = 2.0,
                       local_range: int = 11,
                       query_block_chunk: int = 1,
                       qkv_projection: str = "stock",
                       cache_format: str = "int8",
                       cache_residency_backend: str = "cpu"):
        self.streaming_active = streaming
        self.sampling_mode = sampling_mode
        self.qkv_projection_mode = str(qkv_projection)
        if self.qkv_projection_mode not in (
            "stock", "shared_int8_experimental"
        ):
            raise ValueError(
                f"Unknown FlashVSR QKV projection mode: "
                f"{self.qkv_projection_mode}"
            )
        # Match official FlashVSR v1.1's resolution normalization:
        # topk_ratio = sparse_ratio * 768 * 1280 / (height * width)
        raw_sparse_ratio = float(sparse_ratio)
        reference_pixels = 768 * 1280
        prepared_pixels = max(1, self.video.height * self.video.width)
        self.lcsa_sparse_ratio = (
            raw_sparse_ratio * reference_pixels / prepared_pixels
        )
        self.lcsa_local_range = max(1, int(local_range))
        # Zero selects a conservative value from the free CUDA memory seen by
        # the first self-attention block of each model segment.
        self.lcsa_query_block_chunk = max(0, int(query_block_chunk))
        self.resolved_lcsa_query_block_chunk = {}
        self.seen_self_attention_blocks.clear()
        faithful_mode = (
            sampling_mode
            if sampling_mode in (
                "streaming_faithful_full",
                "streaming_faithful_lowvram",
            )
            else None
        )
        self.kv_cache.configure(
            faithful_mode,
            self.total_wan_blocks,
            cache_format=cache_format,
            residency_backend=cache_residency_backend,
        )
        if streaming:
            capability = (
                "available" if self.int8_qkv_format is not None
                else "unavailable"
            )
            if self.profile_enabled:
                print(
                    "[FlashVSR profiler] LCSA sparse ratio: "
                    f"raw={raw_sparse_ratio:g}, "
                    f"effective={self.lcsa_sparse_ratio:.6g} at "
                    f"{self.video.width}x{self.video.height} "
                    "(official 768x1280 normalization)."
                )
            print(
                f"[FlashVSR] QKV projection={self.qkv_projection_mode} "
                f"(experimental shared path {capability})."
            )

    def begin_model_chunk(self):
        self.seen_self_attention_blocks.clear()
        self.resolved_lcsa_query_block_chunk = {}
        self.kv_cache.begin_chunk(
            initial=(
                self.kv_cache.enabled
                and self.current_rope_start == 0
                and self.current_latent_frames == 6
            )
        )

    def end_model_chunk(self):
        self.kv_cache.end_chunk()

    def streaming_attention(self, q, k, v, heads, mask=None,
                            attn_precision=None, skip_reshape=False,
                            skip_output_reshape=False,
                            transformer_options=None, **kwargs):
        options = transformer_options or {}
        block_index = int(options.get("block_index", -1))
        spatial_tokens = (
            (self.video.height // 16) * (self.video.width // 16)
        )
        expected_tokens = self.current_latent_frames * spatial_tokens

        # Wan invokes self-attention before cross-attention in every block.
        # The seen-block guard avoids misidentifying cross-attention at rare
        # resolutions where its text length equals the video token count.
        is_self_attention = (
            block_index >= 0
            and block_index not in self.seen_self_attention_blocks
            and mask is None
            and not skip_reshape
            and q.ndim == 3
            and k.ndim == 3
            and v.ndim == 3
            and q.shape[1] == expected_tokens
            and k.shape[1] == expected_tokens
            and v.shape[1] == expected_tokens
        )
        if not is_self_attention:
            return self.call_attention_backend(
                q, k, v, heads, mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                transformer_options=options,
                **kwargs,
            )

        self.seen_self_attention_blocks.add(block_index)
        tokens_per_frame = spatial_tokens
        cache_active = self.kv_cache.participates(block_index)
        current_k = k
        current_v = v
        compact_descriptor = None
        if cache_active:
            use_compact_sparge = (
                not self.kv_cache.initial_chunk
                and self.kv_cache.cache_format == "int8"
                and getattr(
                    self.sparse_attention_backend,
                    "flashvsr_block_sparse", False,
                )
            )
            if use_compact_sparge:
                compact_descriptor = self.kv_cache.describe(
                    block_index, k, v, tokens_per_frame, heads
                )
            if compact_descriptor is None:
                cache_marker = self.profile_start(k)
                k, v = self.kv_cache.stage(
                    block_index, k, v, tokens_per_frame
                )
                self.profile_end("kv_cache_stage_h2d", cache_marker)
        attention_marker = self.profile_start(q) if cache_active else None
        if compact_descriptor is not None:
            attended = self._streaming_lcsa_attention_compact(
                block_index, q, compact_descriptor, heads,
                transformer_options=options,
            )
        else:
            attended = self._streaming_lcsa_attention(
                block_index, q, k, v, heads,
                attn_precision=attn_precision,
                transformer_options=options,
                **kwargs,
            )
        self.profile_end("kv_cached_attention", attention_marker)
        if cache_active:
            cache_marker = self.profile_start(current_k)
            self.kv_cache.commit(
                block_index, current_k, current_v, tokens_per_frame, heads
            )
            self.profile_end("kv_cache_commit_enqueue", cache_marker)
        return attended

    def _local_spatial_mask(self, grid_h, grid_w, local_range, device):
        device = torch.device(device)
        cache_key = (grid_h, grid_w, local_range, device.type, device.index)
        cached = self.local_spatial_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        rows = torch.arange(grid_h, device=device)
        cols = torch.arange(grid_w, device=device)
        row, col = torch.meshgrid(rows, cols, indexing="ij")
        q_row = row.reshape(-1, 1)
        q_col = col.reshape(-1, 1)
        k_row = row.reshape(1, -1)
        k_col = col.reshape(1, -1)
        half = local_range // 2
        mask = (
            (k_row >= q_row - half)
            & (k_row <= q_row - half + local_range - 1)
            & (k_col >= q_col - half)
            & (k_col <= q_col - half + local_range - 1)
        )
        self.local_spatial_mask_cache[cache_key] = mask
        return mask

    def _local_block_topology(self, q_temporal, k_temporal, grid_h, grid_w,
                              local_range, device):
        """Return a cached, broadcastable LCSA local-neighborhood mask."""
        device = torch.device(device)
        cache_key = (
            q_temporal, k_temporal, grid_h, grid_w, local_range,
            device.type, device.index,
        )
        cached = self.local_topology_cache.get(cache_key)
        if cached is not None:
            return cached

        spatial = grid_h * grid_w
        local = self._local_spatial_mask(
            grid_h, grid_w, local_range, device
        )
        # This reshape of an expanded tensor may materialize. Cache the small
        # static result once instead of rebuilding it in every Wan block.
        topology = (
            local.view(1, 1, 1, spatial, 1, spatial)
            .expand(
                1, 1, q_temporal, spatial,
                k_temporal, spatial,
            )
            .reshape(
                1, 1, q_temporal, spatial,
                k_temporal * spatial,
            )
            .contiguous()
        )
        # Calculate the static count on the CPU to avoid synchronizing CUDA
        # merely to inspect the cached boolean topology.
        half = local_range // 2
        spatial_pairs = 0
        for row in range(grid_h):
            row_count = max(
                0,
                min(grid_h, row - half + local_range)
                - max(0, row - half),
            )
            for col in range(grid_w):
                col_count = max(
                    0,
                    min(grid_w, col - half + local_range)
                    - max(0, col - half),
                )
                spatial_pairs += row_count * col_count
        eligible_per_temporal_query = spatial_pairs * k_temporal
        cached = (topology, eligible_per_temporal_query)
        self.local_topology_cache[cache_key] = cached
        return cached

    def _resolve_query_block_chunk(
        self, q_blocks, batch, heads, key_tokens, channels, q
    ):
        requested = self.lcsa_query_block_chunk
        if requested > 0:
            return min(requested, q_blocks)
        cached_resolution = self.resolved_lcsa_query_block_chunk.get(
            int(key_tokens)
        )
        if cached_resolution is not None:
            return min(cached_resolution, q_blocks)

        resolved = 1
        free_bytes = None
        if q.device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(q.device)
                # A generic masked backend may hold both an expanded mask and
                # float attention workspace. Budget eight bytes per Q/K pair,
                # add two query-output buffers, and use at most 40% of memory
                # still free after Q/K/V have been created.
                pairs_per_block = batch * heads * 128 * key_tokens
                output_per_block = (
                    batch * 128 * channels * q.element_size()
                )
                bytes_per_block = pairs_per_block * 8 + output_per_block * 2
                affordable = max(
                    1, int(free_bytes * 0.40) // max(1, bytes_per_block)
                )
                for candidate in (1, 2, 4, 8, 16, 32):
                    if candidate <= affordable and candidate <= q_blocks:
                        resolved = candidate
            except (RuntimeError, TypeError):
                resolved = 1

        self.resolved_lcsa_query_block_chunk[int(key_tokens)] = resolved
        report_key = (self.current_latent_frames, key_tokens, resolved)
        if report_key not in self.reported_auto_chunks:
            memory = (
                f", free VRAM {free_bytes / (1024 ** 2):.0f} MiB"
                if free_bytes is not None else ""
            )
            print(
                "[FlashVSR] auto query_block_chunk="
                f"{resolved} for {self.current_latent_frames} latent frames"
                f"{memory}."
            )
            self.reported_auto_chunks.add(report_key)
        return min(resolved, q_blocks)

    def _lcsa_token_indices(self, frames, height, width, device):
        """Map 2x8x8 block order to stock Wan token order."""
        device = torch.device(device)
        cache_key = (
            frames, height, width, device.type, device.index
        )
        cached = self.lcsa_token_index_cache.get(cache_key)
        if cached is not None:
            return cached

        temporal = frames // 2
        grid_h = height // 8
        grid_w = width // 8
        token_count = frames * height * width
        block_to_original = (
            torch.arange(token_count, device=device)
            .view(temporal, 2, grid_h, 8, grid_w, 8)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1)
        )
        original_to_key_block = torch.empty_like(block_to_original)
        original_to_key_block[block_to_original] = (
            torch.arange(token_count, device=device) // 128
        )
        cached = (block_to_original, original_to_key_block)
        self.lcsa_token_index_cache[cache_key] = cached
        return cached

    def _lcsa_block_mask(self, q, k, heads, q_frames, k_frames,
                         height, width,
                         grid_h, grid_w):
        """Build FlashVSR's logical 128-query x 128-key topology."""
        batch, q_tokens, channels = q.shape
        k_batch, k_tokens, k_channels = k.shape
        q_temporal = q_frames // 2
        k_temporal = k_frames // 2
        spatial = grid_h * grid_w
        expected_q_tokens = q_frames * height * width
        expected_k_tokens = k_frames * height * width
        if (
            k_batch != batch
            or q_tokens != expected_q_tokens
            or k_tokens != expected_k_tokens
            or k_channels != channels
            or channels % heads
            or q_frames % 2
            or k_frames % 2
        ):
            raise RuntimeError("Invalid FlashVSR LCSA Q/K window shapes.")
        head_dim = channels // heads

        # Pool directly from stock Wan's frame-major token layout. This avoids
        # materializing full contiguous 2x8x8 Q and K window tensors merely to
        # compute one routing vector per block.
        q_pool = q.view(
            batch, q_temporal, 2, grid_h, 8, grid_w, 8,
            heads, head_dim,
        ).mean(dim=(2, 4, 6)).reshape(
            batch, q_temporal, spatial, heads, head_dim
        )
        k_pool = k.view(
            batch, k_temporal, 2, grid_h, 8, grid_w, 8,
            heads, head_dim,
        ).mean(dim=(2, 4, 6)).reshape(
            batch, k_temporal, spatial, heads, head_dim
        )
        return self._lcsa_mask_from_pools(
            q_pool, k_pool, grid_h, grid_w
        )

    def _lcsa_mask_from_pools(self, q_pool, k_pool, grid_h, grid_w):
        """Select FlashVSR block pairs from already-pooled Q/K summaries."""
        batch, q_temporal, spatial, heads, head_dim = q_pool.shape
        k_temporal = k_pool.shape[1]
        if (
            k_pool.shape[0] != batch
            or k_pool.shape[2] != spatial
            or k_pool.shape[3] != heads
            or k_pool.shape[4] != head_dim
        ):
            raise RuntimeError("Invalid FlashVSR cached LCSA K summary.")
        scores = torch.einsum(
            "btnhd,bsmhd->bhtnsm", q_pool, k_pool
        ) / math.sqrt(head_dim)
        scores = scores.reshape(
            batch, heads, q_temporal, spatial,
            k_temporal * spatial,
        ).float()

        local, eligible_count = self._local_block_topology(
            q_temporal, k_temporal, grid_h, grid_w,
            self.lcsa_local_range, scores.device,
        )
        scores.masked_fill_(~local, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)

        # FlashVSR selects block pairs globally for every temporal query
        # window and attention head after locally masked softmax scoring.
        flat = probabilities.reshape(batch, heads, q_temporal, -1)
        requested = int(spatial * spatial * self.lcsa_sparse_ratio) - 1
        selected_count = min(max(1, requested), flat.shape[-1] - 1)
        if selected_count >= eligible_count:
            # Once the budget covers every locally eligible pair, top-k can
            # only establish a zero threshold. Preserve float underflow
            # behavior by selecting positive probabilities rather than
            # returning the geometric mask unconditionally.
            selected = probabilities > 0
            best = probabilities.argmax(dim=-1, keepdim=True)
            selected.scatter_(-1, best, True)
            return selected
        # Only the smallest member of the selected top-k set is needed as the
        # threshold. Keep sorting disabled, then reduce the returned values.
        # This restores the proven baseline path after kthvalue did not offer
        # a sufficiently compelling cross-version advantage.
        top_values = torch.topk(
            flat,
            k=selected_count + 1,
            dim=-1,
            sorted=False,
        ).values
        threshold = top_values.amin(dim=-1, keepdim=True)
        selected = (flat > threshold).view_as(probabilities)

        # Setting every row's best key is equivalent to the former conditional
        # fallback: in non-empty rows the maximum is already above threshold;
        # in empty rows it becomes the required single safe key. This avoids
        # allocating row_has_key, a full safe mask, and torch.where output.
        best = probabilities.argmax(dim=-1, keepdim=True)
        selected.scatter_(-1, best, True)
        return selected

    def _streaming_lcsa_attention_compact(
        self, block_index, q, descriptor, heads, transformer_options=None
    ):
        """Route from cached K summaries and materialize only final HND K/V."""
        height = self.video.height // 16
        width = self.video.width // 16
        grid_h = height // 8
        grid_w = width // 8
        spatial = grid_h * grid_w
        q_frames = self.current_latent_frames
        q_temporal = q_frames // 2
        batch, q_tokens, channels = q.shape
        head_dim = channels // heads
        if q_frames != self.kv_cache.SLOT_FRAMES:
            raise RuntimeError(
                "Compact faithful cache expected a two-frame continuation."
            )

        routing_marker = self.profile_start(q)
        q_pool = q.view(
            batch, q_temporal, 2, grid_h, 8, grid_w, 8,
            heads, head_dim,
        ).mean(dim=(2, 4, 6)).reshape(
            batch, q_temporal, spatial, heads, head_dim
        )
        current_k = descriptor.current_k
        current_pool = current_k.view(
            batch, 1, 2, grid_h, 8, grid_w, 8,
            heads, head_dim,
        ).mean(dim=(2, 4, 6)).reshape(
            batch, 1, spatial, heads, head_dim
        )
        summary_marker = self.profile_start(q)
        cached_pools = tuple(
            summary.to(
                device=q.device, dtype=q.dtype, non_blocking=False
            )
            for _cached_k, _cached_v, summary in descriptor.slots
        )
        self.profile_end("kv_summary_h2d", summary_marker)
        k_pool = torch.cat((*cached_pools, current_pool), dim=1)
        block_mask = self._lcsa_mask_from_pools(
            q_pool, k_pool, grid_h, grid_w
        ).reshape(batch, heads, q_temporal * spatial, -1)
        self.profile_end("lcsa_routing", routing_marker)

        q_block_to_original, _ = self._lcsa_token_indices(
            q_frames, height, width, q.device
        )
        slot_block_to_original, _ = self._lcsa_token_indices(
            self.kv_cache.SLOT_FRAMES, height, width, q.device
        )
        try:
            return self.sparse_attention_backend.run_flashvsr_compact(
                q,
                descriptor,
                self.kv_cache,
                heads,
                block_mask,
                q_block_to_original,
                slot_block_to_original,
                profiler=self,
            )
        except Exception as error:
            raise RuntimeError(
                "FlashVSR compact-cache Sparge Attention failed. Confirm "
                "that the SpargeAttn wheel matches this ComfyUI Python, "
                "PyTorch, CUDA, and GPU architecture."
            ) from error

    def _streaming_lcsa_attention(self, block_index, q, k, v, heads,
                                  attn_precision=None,
                                  transformer_options=None, **kwargs):
        height = self.video.height // 16
        width = self.video.width // 16
        q_frames = self.current_latent_frames
        tokens_per_frame = height * width
        if k.shape[1] % tokens_per_frame:
            raise RuntimeError(
                "FlashVSR cached K length is not divisible by the Wan "
                "spatial token count."
            )
        k_frames = k.shape[1] // tokens_per_frame
        grid_h = height // 8
        grid_w = width // 8

        routing_marker = self.profile_start(q)
        block_mask = self._lcsa_block_mask(
            q, k, heads, q_frames, k_frames,
            height, width, grid_h, grid_w
        )
        self.profile_end("lcsa_routing", routing_marker)
        batch, _, channels = q.shape
        q_temporal = q_frames // 2
        spatial = grid_h * grid_w
        q_blocks = q_temporal * spatial
        block_mask = block_mask.reshape(
            batch, heads, q_blocks, -1
        )
        q_block_to_original, _ = self._lcsa_token_indices(
            q_frames, height, width, q.device
        )
        k_block_to_original, original_to_key_block = (
            self._lcsa_token_indices(
                k_frames, height, width, q.device
            )
        )

        # The optional Sparge backend consumes the logical 128-query x 128-key
        # FlashVSR block mask through a private route. It converts that mask to
        # the GPU kernel's physical geometry, gathers directly into final HND
        # layout, and preserves ModelAttentionBackend for all other attention.
        if getattr(
            self.sparse_attention_backend,
            "flashvsr_block_sparse",
            False,
        ):
            try:
                return self.sparse_attention_backend.run_flashvsr(
                    q,
                    k,
                    v,
                    heads,
                    block_mask,
                    q_block_to_original,
                    k_block_to_original,
                    profiler=self,
                )
            except Exception as error:
                raise RuntimeError(
                    "FlashVSR Sparge Attention failed. Confirm that the "
                    "SpargeAttn wheel matches this ComfyUI Python, PyTorch, "
                    "CUDA, and GPU architecture."
                ) from error

        # K/V stay in stock Wan order. Only the active Q slice is gathered into
        # block order, and its result is scattered directly back into Wan
        # order. This removes full contiguous window copies of Q, K, and V.
        attended = torch.empty_like(q)
        chunk = self._resolve_query_block_chunk(
            q_blocks, batch, heads, k.shape[1], channels, q
        )
        for start in range(0, q_blocks, chunk):
            end = min(start + chunk, q_blocks)
            query_indices = q_block_to_original[
                start * 128:end * 128
            ]
            q_chunk = q.index_select(1, query_indices)
            # Resolve key blocks before expanding query rows. A single LCSA
            # query block has one mask shared by all 128 of its tokens, so the
            # singleton query row can be broadcast by ComfyUI's standard
            # masked-attention API instead of allocating 128 identical rows.
            token_mask = block_mask[:, :, start:end].index_select(
                -1, original_to_key_block
            )
            if end - start > 1:
                token_mask = token_mask.repeat_interleave(
                    128,
                    dim=-2,
                    output_size=(end - start) * 128,
                )
            try:
                current = self.call_attention_backend(
                    q_chunk, k, v, heads,
                    mask=token_mask,
                    attn_precision=attn_precision,
                    skip_reshape=False,
                    skip_output_reshape=False,
                    transformer_options=transformer_options,
                    **kwargs,
                )
            except Exception as error:
                raise RuntimeError(
                    "The selected attention backend could not execute "
                    "FlashVSR's per-head LCSA mask. Use ModelAttentionBackend "
                    "with PyTorch attention, another mask-capable backend, "
                    "or switch the sampler to full_video_dense."
                ) from error
            attended.index_copy_(1, query_indices, current)
            del current, token_mask, q_chunk

        return attended

    def reset(self):
        self.current_lq = None
        self.lq_overlap_tail = None
        self.lq_overlap_compact = False
        self.lq_segment_buffers = None
        self.current_latent_frames = 0
        self.current_rope_start = 0
        self.streaming_active = False
        self.sampling_mode = None
        self.resolved_lcsa_query_block_chunk = {}
        self.seen_self_attention_blocks.clear()
        self.int8_qkv_run_active = False
        self.int8_qkv_disabled_reason = None
        self.int8_qkv_capable = self.int8_qkv_format is not None
        self.kv_cache.clear()
        self.lq.model.clear_cache()

    def cleanup(self):
        self.reset()

    def block0_patch(self, args, extra):
        if self.current_lq is None:
            raise RuntimeError(
                "FlashVSR LQ tokens were not prepared for sampling."
            )
        lq_tokens = self.current_lq[0].to(
            device=args["img"].device,
            dtype=args["img"].dtype,
        )
        if lq_tokens.shape != args["img"].shape:
            raise RuntimeError(
                "FlashVSR LQ token shape does not match Wan image tokens: "
                f"{tuple(lq_tokens.shape)} != {tuple(args['img'].shape)}"
            )
        if self._profile_first_prefill_active():
            # args["img"] is stock Wan's patch-embedding output immediately
            # before FlashVSR's sole LQ injection point.
            self.profile_activation("patch_tokens", args["img"])
            self.profile_activation("lq_tokens", lq_tokens)

        patched = dict(args)
        strength = self.conditioning_strength
        if strength == 0.0:
            patched["img"] = args["img"]
        else:
            # Block 0 is the sole consumer of this image-token value. Inject
            # conditioning into its existing allocation instead of creating a
            # second complete Wan token tensor for every streaming segment.
            # alpha= also avoids materializing lq_tokens * strength.
            args["img"].add_(lq_tokens, alpha=strength)
            patched["img"] = args["img"]
        if self._profile_first_prefill_active():
            self.profile_activation("patch_plus_lq", patched["img"])

        result = extra["original_block"](patched)
        if (
            self._profile_first_prefill_active()
            and isinstance(result, dict)
            and torch.is_tensor(result.get("img"))
        ):
            self.profile_activation("block0_output", result["img"])
        return result

    def _prepare_lq_chunk(
        self, process_index: int, new_latent_frames: int = 2
    ):
        device = self.lq.patcher.load_device
        dtype = self.lq.compute_dtype
        video = self.video.tensor
        output_buffers = None
        output_offsets = None
        spatial_tokens = (
            (self.video.height // 16) * (self.video.width // 16)
        )
        expected_frames = (
            6 if process_index == 0 else int(new_latent_frames)
        )
        expected_tokens = expected_frames * spatial_tokens

        if process_index == 0:
            clips = [
                video[:, :, max(0, i * 4 - 3):(i + 1) * 4 - 3]
                for i in range(7)
            ]
        else:
            base = process_index * 8 + 17
            clips = [
                video[:, :, base + i * 4:base + i * 4 + 4]
                for i in range(int(new_latent_frames))
            ]

        if self.profile_enabled:
            states = []
            for name in ("conv1", "conv2"):
                module = getattr(self.lq.model, name)
                weight = module.weight
                states.append(
                    f"{name}={weight.device}/{weight.dtype}/"
                    f"{type(module).__name__}"
                )
            linear = self.lq.model.linear_layers[0]
            states.append(
                f"linear={linear.weight.device}/{linear.weight.dtype}/"
                f"{type(linear).__name__}"
            )
            print(
                f"[FlashVSR profiler] LQ process_index={process_index}, "
                f"new_latent_frames={int(new_latent_frames)}: "
                + ", ".join(states)
            )

        for clip in clips:
            transfer_marker = self.profile_start(device)
            device_clip = clip.to(device=device, dtype=dtype)
            self.profile_end("lq_input_transfer", transfer_marker)
            current = self.lq.model.stream_forward(
                device_clip,
                profiler=self if self.profile_enabled else None,
            )
            if current is None:
                continue
            if output_buffers is None:
                output_buffers = [
                    new.new_empty(
                        (new.shape[0], expected_tokens, new.shape[2])
                    )
                    for new in current
                ]
                output_offsets = [0] * len(current)
            if len(output_buffers) != len(current):
                raise RuntimeError(
                    "LQ projector changed its output-layer count while "
                    "streaming."
                )
            for index, new in enumerate(current):
                if (
                    new.shape[0] != output_buffers[index].shape[0]
                    or new.shape[2] != output_buffers[index].shape[2]
                ):
                    raise RuntimeError(
                        "LQ projector changed its token shape while "
                        "streaming."
                    )
                end = output_offsets[index] + new.shape[1]
                if end > expected_tokens:
                    raise RuntimeError(
                        "LQ projector produced more conditioning tokens "
                        "than expected."
                    )
                output_buffers[index][
                    :, output_offsets[index]:end
                ].copy_(new)
                output_offsets[index] = end
        if output_buffers is None:
            raise RuntimeError("LQ projector produced no conditioning tokens.")
        if any(offset != expected_tokens for offset in output_offsets):
            raise RuntimeError(
                "LQ projector produced an unexpected conditioning-token "
                "count."
            )
        return output_buffers

    def _assemble_lq_segment(self, overlap, current):
        """Copy overlap and new tokens into reusable segment allocations."""
        if len(overlap) != len(current):
            raise RuntimeError(
                "LQ projector overlap-layer count does not match the "
                "current output."
            )
        required_tokens = overlap[0].shape[1] + current[0].shape[1]
        reusable = self.lq_segment_buffers
        if reusable is None or len(reusable) != len(current):
            reusable = [None] * len(current)

        assembled = []
        for index, (old, new) in enumerate(zip(overlap, current)):
            if (
                old.shape[0] != new.shape[0]
                or old.shape[2] != new.shape[2]
            ):
                raise RuntimeError(
                    "LQ projector overlap-token shape does not match the "
                    "current output."
                )
            needed = old.shape[1] + new.shape[1]
            if needed != required_tokens:
                raise RuntimeError(
                    "LQ projector layers produced inconsistent token counts."
                )
            buffer = reusable[index]
            if (
                buffer is None
                or buffer.shape[0] != new.shape[0]
                or buffer.shape[1] < needed
                or buffer.shape[2] != new.shape[2]
                or buffer.device != new.device
                or buffer.dtype != new.dtype
            ):
                buffer = new.new_empty(
                    (new.shape[0], needed, new.shape[2])
                )
                reusable[index] = buffer
            segment = buffer[:, :needed]
            segment[:, :old.shape[1]].copy_(old)
            segment[:, old.shape[1]:].copy_(new)
            assembled.append(segment)

        self.lq_segment_buffers = reusable
        return assembled

    def prepare_lq_for_process(
        self, process_index: int, new_latent_frames: int = 2
    ):
        current = self._prepare_lq_chunk(
            process_index, new_latent_frames
        )
        spatial_tokens = (
            (self.video.height // 16) * (self.video.width // 16)
        )
        tail_tokens = 2 * spatial_tokens
        if process_index == 0:
            self.current_lq = current
            self.lq_overlap_tail = [
                # current_lq already owns this complete allocation during the
                # first Wan call, so a view avoids duplicating two frames.
                layer[:, -tail_tokens:].detach()
                for layer in current
            ]
            self.lq_overlap_compact = False
            self.current_latent_frames = 6
            self.current_rope_start = 0
        else:
            if self.lq_overlap_tail is None:
                raise RuntimeError(
                    "FlashVSR overlap conditioning was not initialized."
                )
            self.current_lq = self._assemble_lq_segment(
                self.lq_overlap_tail, current
            )
            next_tail = []
            for old, layer in zip(self.lq_overlap_tail, current):
                source = layer[:, -tail_tokens:].detach()
                if self.lq_overlap_compact and old.shape == source.shape:
                    # Reuse the compact two-frame buffer after torch.cat has
                    # consumed its previous contents on the current stream.
                    old.copy_(source)
                    next_tail.append(old)
                else:
                    # The initial tail is a view into the six-frame projector
                    # output. Replace it with an owning two-frame allocation.
                    next_tail.append(source.clone())
            self.lq_overlap_tail = next_tail
            self.lq_overlap_compact = True
            self.current_latent_frames = 2 + int(new_latent_frames)
            self.current_rope_start = 2 + process_index * 2

    def prepare_lq_for_faithful_process(
        self, process_index: int, new_latent_frames: int = 2
    ):
        """Prepare paper-layout LQ tokens without replaying overlap frames."""
        if int(new_latent_frames) != 2 and process_index != 0:
            raise ValueError(
                "Faithful FlashVSR continuations require two new latent "
                "frames."
            )
        current = self._prepare_lq_chunk(
            process_index, new_latent_frames
        )
        self.current_lq = current
        self.lq_overlap_tail = None
        self.lq_overlap_compact = False
        self.lq_segment_buffers = None
        if process_index == 0:
            self.current_latent_frames = 6
            self.current_rope_start = 0
        else:
            self.current_latent_frames = int(new_latent_frames)
            # The first six-frame prefill occupies positions 0..5. Each
            # subsequent process contributes exactly two new positions.
            self.current_rope_start = 4 + process_index * 2

    def prepare_lq_full(self, latent_frames: int):
        process_total = latent_frames // 2 - 2
        if process_total < 1 or latent_frames % 2:
            raise ValueError(
                "FlashVSR requires an even latent length of at least six."
            )

        self.lq.model.clear_cache()
        chunks = []
        for process_index in range(process_total):
            current = self._prepare_lq_chunk(process_index)
            chunks.append(current[0])

        tokens = torch.cat(chunks, dim=1)
        expected_tokens = (
            latent_frames
            * (self.video.height // 16)
            * (self.video.width // 16)
        )
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "FlashVSR LQ projector produced an unexpected token count: "
                f"{tokens.shape[1]} != {expected_tokens}"
            )
        self.current_lq = [tokens]
        self.current_latent_frames = latent_frames
        self.current_rope_start = 0


class _FlashVSRBlock0Patch:
    """Block patch that declares its managed LQ model dependency.

    ComfyUI discovers ``models()`` dependencies directly from model patches
    inside ``load_models_gpu``. Retaining the additional-model registration as
    well supports the ordinary sampling-preparation path and older ComfyUI
    versions; its loader de-duplicates the same patcher.
    """

    def __init__(self, runtime: FlashVSRRuntime):
        self.runtime = runtime

    def __call__(self, args, extra):
        options = args.get("transformer_options") or {}
        runtime = options.get(ACTIVE_RUNTIME_OPTION, self.runtime)
        if not isinstance(runtime, FlashVSRRuntime):
            runtime = self.runtime
        return runtime.block0_patch(args, extra)

    def models(self):
        return [self.runtime.lq.patcher]


def patch_model(model, runtime: FlashVSRRuntime):
    patched = model.clone()
    blocks = None
    try:
        blocks = patched.get_model_object("diffusion_model.blocks")
        runtime.set_total_wan_blocks(len(blocks))
    except (AttributeError, KeyError, TypeError, RuntimeError):
        try:
            blocks = patched.model.diffusion_model.blocks
            runtime.set_total_wan_blocks(len(blocks))
        except (AttributeError, TypeError, RuntimeError):
            runtime.set_total_wan_blocks(0)
    try:
        runtime.set_dynamic_vram(patched.is_dynamic())
    except (AttributeError, TypeError, RuntimeError):
        runtime.set_dynamic_vram(False)
    if blocks is not None:
        install_wan_qkv_patches(patched, runtime, blocks)
        install_streamed_wan_block_patches(patched, runtime, blocks)
    else:
        runtime.configure_int8_qkv(None)
    patched.set_model_patch_replace(
        _FlashVSRBlock0Patch(runtime), "dit", "double_block", 0
    )
    patched.set_additional_models("flashvsr_lq", [runtime.lq.patcher])
    transformer_options = patched.model_options.setdefault(
        "transformer_options", {}
    )
    runtime.set_attention_backend(
        transformer_options.get("optimized_attention_override")
    )
    runtime.set_sparse_attention_backend(
        transformer_options.get(SPARSE_BACKEND_OPTION)
    )
    patched.set_model_optimized_attention(runtime.attention_override)
    runtime.set_installed_attention_override(
        patched.model_options["transformer_options"].get(
            "optimized_attention_override"
        )
    )
    return patched
