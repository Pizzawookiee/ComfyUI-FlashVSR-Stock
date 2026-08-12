from __future__ import annotations

from dataclasses import dataclass
import math

import torch

import comfy.ldm.modules.attention as comfy_attention

from .components import ComponentHandle
from .sparse_backend import SPARSE_BACKEND_OPTION


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

        if not runtime.streaming_active:
            return runtime.call_attention_backend(
                q, k, v, heads, mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                transformer_options=options,
                **kwargs,
            )

        return runtime.streaming_attention(
            q, k, v, heads, mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            transformer_options=options,
            **kwargs,
        )


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
        self.lcsa_sparse_ratio = 2.0
        self.lcsa_local_range = 11
        self.lcsa_query_block_chunk = 1
        self.resolved_lcsa_query_block_chunk = None
        self.local_spatial_mask_cache = {}
        self.local_topology_cache = {}
        self.lcsa_token_index_cache = {}
        self.reported_auto_chunks = set()
        self.seen_self_attention_blocks = set()
        self.attention_backend = None
        self.sparse_attention_backend = None
        self.attention_override = StreamingAttentionDispatcher(self)
        self.installed_attention_override = None
        self.profile_enabled = False
        self.profile_events = {}
        self.profile_devices = set()

    def begin_profile(self, enabled: bool):
        """Start a non-synchronizing CUDA-event profiling collection."""
        self.profile_enabled = bool(enabled and torch.cuda.is_available())
        self.profile_events = {}
        self.profile_devices = set()
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
            "lcsa_routing",
            "sparge_layout_qkv",
            "sparge_mask_convert",
            "sparge_kernel",
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

        nested = sum(
            totals.get(name, 0.0) for name in (
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

    def begin_sampling(self, streaming: bool, sparse_ratio: float = 2.0,
                       local_range: int = 11,
                       query_block_chunk: int = 1):
        self.streaming_active = streaming
        self.lcsa_sparse_ratio = max(0.1, float(sparse_ratio))
        self.lcsa_local_range = max(1, int(local_range))
        # Zero selects a conservative value from the free CUDA memory seen by
        # the first self-attention block of each model segment.
        self.lcsa_query_block_chunk = max(0, int(query_block_chunk))
        self.resolved_lcsa_query_block_chunk = None
        self.seen_self_attention_blocks.clear()

    def begin_model_chunk(self):
        self.seen_self_attention_blocks.clear()
        self.resolved_lcsa_query_block_chunk = None

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
        return self._streaming_lcsa_attention(
            block_index, q, k, v, heads,
            attn_precision=attn_precision,
            transformer_options=options,
            **kwargs,
        )

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

    def _local_block_topology(self, q_temporal, grid_h, grid_w,
                              local_range, device):
        """Return a cached, broadcastable LCSA local-neighborhood mask."""
        device = torch.device(device)
        cache_key = (
            q_temporal, grid_h, grid_w, local_range,
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
            local.view(1, 1, 1, spatial, 1, spatial, 1)
            .expand(
                1, 1, q_temporal, spatial,
                q_temporal, spatial, 2,
            )
            .reshape(
                1, 1, q_temporal, spatial,
                q_temporal * spatial * 2,
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
        eligible_per_temporal_query = (
            spatial_pairs * q_temporal * 2
        )
        cached = (topology, eligible_per_temporal_query)
        self.local_topology_cache[cache_key] = cached
        return cached

    def _resolve_query_block_chunk(
        self, q_blocks, batch, heads, key_tokens, channels, q
    ):
        requested = self.lcsa_query_block_chunk
        if requested > 0:
            return min(requested, q_blocks)
        if self.resolved_lcsa_query_block_chunk is not None:
            return min(self.resolved_lcsa_query_block_chunk, q_blocks)

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

        self.resolved_lcsa_query_block_chunk = resolved
        report_key = (self.current_latent_frames, resolved)
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
            torch.arange(token_count, device=device) // 64
        )
        cached = (block_to_original, original_to_key_block)
        self.lcsa_token_index_cache[cache_key] = cached
        return cached

    def _lcsa_block_mask(self, q, k, heads, frames, height, width,
                         grid_h, grid_w):
        """Build FlashVSR's 128-query x 64-key block topology."""
        batch, q_tokens, channels = q.shape
        k_batch, k_tokens, k_channels = k.shape
        q_temporal = frames // 2
        k_temporal = q_temporal
        spatial = grid_h * grid_w
        expected_tokens = frames * height * width
        if (
            k_batch != batch
            or q_tokens != expected_tokens
            or k_tokens != expected_tokens
            or k_channels != channels
            or channels % heads
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
        k_pool = (
            k.view(
                batch, k_temporal, 2, grid_h, 8, grid_w, 8,
                heads, head_dim,
            )
            .mean(dim=(4, 6))
            .permute(0, 1, 3, 4, 2, 5, 6)
            .reshape(
                batch, k_temporal, spatial, 2, heads, head_dim
            )
        )
        scores = torch.einsum(
            "btnhd,bsmrhd->bhtnsmr", q_pool, k_pool
        ) / math.sqrt(head_dim)
        scores = scores.reshape(
            batch, heads, q_temporal, spatial,
            k_temporal * spatial * 2,
        ).float()

        local, eligible_count = self._local_block_topology(
            q_temporal, grid_h, grid_w,
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

    def _streaming_lcsa_attention(self, block_index, q, k, v, heads,
                                  attn_precision=None,
                                  transformer_options=None, **kwargs):
        height = self.video.height // 16
        width = self.video.width // 16
        frames = self.current_latent_frames
        grid_h = height // 8
        grid_w = width // 8

        routing_marker = self.profile_start(q)
        block_mask = self._lcsa_block_mask(
            q, k, heads, frames, height, width, grid_h, grid_w
        )
        self.profile_end("lcsa_routing", routing_marker)
        batch, _, channels = q.shape
        q_temporal = frames // 2
        spatial = grid_h * grid_w
        q_blocks = q_temporal * spatial
        block_mask = block_mask.reshape(
            batch, heads, q_blocks, -1
        )
        block_to_original, original_to_key_block = (
            self._lcsa_token_indices(
                frames, height, width, q.device
            )
        )

        # The optional Sparge backend consumes the native 128-query x 64-key
        # FlashVSR block mask through a private route. It gathers directly into
        # its final HND layout, executes one complete sparse layer, and restores
        # stock Wan order without replacing ModelAttentionBackend.
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
                    block_to_original,
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
            query_indices = block_to_original[
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
        self.resolved_lcsa_query_block_chunk = None
        self.seen_self_attention_blocks.clear()
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
        return extra["original_block"](patched)

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
