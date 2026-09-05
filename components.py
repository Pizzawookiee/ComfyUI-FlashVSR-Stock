from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils


CACHE_T = 2


def _component_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if model_management.should_use_bf16(device):
        return torch.bfloat16
    if model_management.should_use_fp16(device):
        return torch.float16
    return torch.float32


def _state_dtype(sd: dict[str, torch.Tensor]) -> torch.dtype:
    for value in sd.values():
        if torch.is_tensor(value) and value.is_floating_point():
            return value.dtype
    return torch.float32


class ManagedComponent(nn.Module):
    def __init__(self, compute_dtype: torch.dtype):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.manual_cast_dtype = compute_dtype

    def get_dtype(self):
        return self.compute_dtype


@dataclass
class ComponentHandle:
    model: ManagedComponent
    patcher: comfy.model_patcher.ModelPatcher
    compute_dtype: torch.dtype

    @property
    def device(self) -> torch.device:
        return self.patcher.load_device

    def cleanup(self):
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()
        if hasattr(self.model, "clean_mem"):
            self.model.clean_mem()


class ChannelRMSNorm(nn.Module):
    """FlashVSR's channel-first RMS normalization.

    This stays custom because torch/comfy RMSNorm normalizes the final axis,
    while FlashVSR normalizes channel axis 1 of a BCFHW tensor.
    """

    def __init__(self, dim: int, device=None, dtype=None):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1, 1), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.to(device=x.device, dtype=x.dtype)
        return F.normalize(x, dim=1) * self.scale * gamma


class PixelUnshuffle3D(nn.Module):
    def __init__(self, temporal: int, height: int, width: int):
        super().__init__()
        self.temporal = temporal
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        remainder = x.shape[2] % self.temporal
        if remainder:
            pad_frames = self.temporal - remainder
            x = torch.cat((x[:, :, :1].repeat(1, 1, pad_frames, 1, 1), x), dim=2)
        return rearrange(
            x,
            "b c (f ft) (h ph) (w pw) -> b (c ft ph pw) f h w",
            ft=self.temporal,
            ph=self.height,
            pw=self.width,
        )


def _causal_conv_class(operations):
    class CausalConv3d(operations.Conv3d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._causal_padding = (
                self.padding[2], self.padding[2],
                self.padding[1], self.padding[1],
                2 * self.padding[0], 0,
            )
            self.padding = (0, 0, 0)

        def prepare_input(
            self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None
        ):
            padding = list(self._causal_padding)
            if cache_x is not None and padding[4] > 0:
                cache_x = cache_x.to(device=x.device, dtype=x.dtype)
                x = torch.cat((cache_x, x), dim=2)
                padding[4] = max(0, padding[4] - cache_x.shape[2])
            return F.pad(x, padding, mode="replicate")

        def forward_prepared(self, x: torch.Tensor):
            return super().forward(x)

        def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None):
            return self.forward_prepared(self.prepare_input(x, cache_x))

    return CausalConv3d

class ChunkedCausalConv3d(nn.Module):
    """Output-channel chunks of one causal Conv3d checkpoint tensor."""

    def __init__(
        self, CausalConv3d, in_channels, out_channels, kernel_size, *,
        chunks, stride, padding, device=None, dtype=None,
    ):
        super().__init__()
        if out_channels % chunks:
            raise ValueError("Chunked causal Conv3d requires even output chunks.")
        self.out_channels = int(out_channels)
        self.chunk_count = int(chunks)
        self.chunk_channels = self.out_channels // self.chunk_count
        self.chunks = nn.ModuleList([
            CausalConv3d(
                in_channels, self.chunk_channels, kernel_size,
                stride=stride, padding=padding, device=device, dtype=dtype,
            )
            for _ in range(self.chunk_count)
        ])

    @property
    def weight(self):
        # Diagnostics only. The real checkpoint weight is split across chunks.
        return self.chunks[0].weight

    @property
    def bias(self):
        return self.chunks[0].bias

    def prepare_input(self, x, cache_x=None):
        return self.chunks[0].prepare_input(x, cache_x)

    def forward_chunk(self, index, prepared):
        return self.chunks[index].forward_prepared(prepared)

    def split_state_dict(self, state_dict, prefix):
        weight = state_dict.pop(f"{prefix}.weight")
        bias = state_dict.pop(f"{prefix}.bias", None)
        for index in range(self.chunk_count):
            start = index * self.chunk_channels
            end = start + self.chunk_channels
            state_dict[f"{prefix}.chunks.{index}.weight"] = (
                weight[start:end].clone()
            )
            if bias is not None:
                state_dict[f"{prefix}.chunks.{index}.bias"] = (
                    bias[start:end].clone()
                )


class ChunkedLinearProjection(nn.Module):
    """Input-channel chunks of one Linear checkpoint projection."""

    def __init__(
        self, operations, in_features, out_features, *, chunks,
        device=None, dtype=None,
    ):
        super().__init__()
        if in_features % chunks:
            raise ValueError("Chunked linear requires even input chunks.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.chunk_count = int(chunks)
        self.chunk_features = self.in_features // self.chunk_count
        self.chunks = nn.ModuleList([
            operations.Linear(
                self.chunk_features, self.out_features,
                bias=(index == 0), device=device, dtype=dtype,
            )
            for index in range(self.chunk_count)
        ])

    @property
    def weight(self):
        # Diagnostics only. The real checkpoint weight is split by columns.
        return self.chunks[0].weight

    def forward_chunk(self, index, x):
        return self.chunks[index](x)

    def split_state_dict(self, state_dict, prefix):
        weight = state_dict.pop(f"{prefix}.weight")
        bias = state_dict.pop(f"{prefix}.bias", None)
        for index in range(self.chunk_count):
            start = index * self.chunk_features
            end = start + self.chunk_features
            state_dict[f"{prefix}.chunks.{index}.weight"] = (
                weight[:, start:end].contiguous()
            )
        if bias is not None:
            state_dict[f"{prefix}.chunks.0.bias"] = bias.clone()



class LQProjector(ManagedComponent):
    # Two stacked 3x3 spatial convolutions give a two-cell receptive-field
    # radius in pixel-unshuffled space. A two-cell halo therefore makes each
    # 2x2 tile equivalent to the untiled projector inside its core.
    SPATIAL_TILE_PARTS = 2
    SPATIAL_TILE_HALO = 2
    CONV2_CHANNEL_CHUNKS = 4
    # Bound the explicit im2col patch matrix. Unlike cuDNN Conv3d workspaces,
    # this is a hard application-level cap and scales safely to large frames.
    CONV2_IM2COL_BUDGET_MIB = 192

    def __init__(self, operations, compute_dtype, device=None, weight_dtype=None):
        super().__init__(compute_dtype)
        CausalConv3d = _causal_conv_class(operations)
        self.pixel_shuffle = PixelUnshuffle3D(1, 16, 16)
        self.conv1 = CausalConv3d(
            3 * 16 * 16, 2048, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1),
            device=device, dtype=weight_dtype,
        )
        self.norm1 = ChannelRMSNorm(2048, device=device, dtype=weight_dtype)
        self.act1 = nn.SiLU()
        self.conv2 = ChunkedCausalConv3d(
            CausalConv3d,
            2048,
            3072,
            (4, 3, 3),
            chunks=self.CONV2_CHANNEL_CHUNKS,
            stride=(2, 1, 1),
            padding=(1, 1, 1),
            device=device,
            dtype=weight_dtype,
        )
        self.norm2 = ChannelRMSNorm(3072, device=device, dtype=weight_dtype)
        self.act2 = nn.SiLU()
        # The released v1.1 checkpoint has one projection and conditions block 0.
        self.linear_layers = nn.ModuleList([
            ChunkedLinearProjection(
                operations,
                3072,
                1536,
                chunks=self.CONV2_CHANNEL_CHUNKS,
                device=device,
                dtype=weight_dtype,
            )
        ])
        self.clear_cache()

    def patch_state_dict(self, state_dict):
        # Split only the checkpoint representation. Runtime modules remain
        # ordinary ComfyUI-managed ops, so Dynamic VRAM/AIMDO sees bounded
        # weights instead of one ~3072-channel Conv3d allocation.
        self.conv2.split_state_dict(state_dict, "conv2")
        for index, layer in enumerate(self.linear_layers):
            layer.split_state_dict(state_dict, f"linear_layers.{index}")
        return state_dict

    def clear_cache(self):
        self.cache = {"conv1": None, "conv2": None}
        self.clip_idx = 0

    def _store_causal_tail(self, key: str, source: torch.Tensor):
        """Retain only the compact causal tail and reuse it when possible."""
        tail = source[:, :, -CACHE_T:].detach()
        stored = self.cache[key]
        if (
            stored is not None
            and stored.shape == tail.shape
            and stored.device == tail.device
            and stored.dtype == tail.dtype
        ):
            stored.copy_(tail)
        else:
            # A detached slice would retain the complete source allocation.
            # Clone exactly the two causal frames into an owning buffer.
            self.cache[key] = tail.clone()

    @classmethod
    def _spatial_tiles(cls, height: int, width: int):
        parts_y = min(cls.SPATIAL_TILE_PARTS, max(1, int(height)))
        parts_x = min(cls.SPATIAL_TILE_PARTS, max(1, int(width)))
        halo = cls.SPATIAL_TILE_HALO
        for tile_y in range(parts_y):
            y0 = height * tile_y // parts_y
            y1 = height * (tile_y + 1) // parts_y
            for tile_x in range(parts_x):
                x0 = width * tile_x // parts_x
                x1 = width * (tile_x + 1) // parts_x
                ey0 = max(0, y0 - halo)
                ey1 = min(height, y1 + halo)
                ex0 = max(0, x0 - halo)
                ex1 = min(width, x1 + halo)
                yield (
                    y0, y1, x0, x1,
                    ey0, ey1, ex0, ex1,
                    y0 - ey0, x0 - ex0,
                )

    @staticmethod
    def _cache_tile(cache, ey0, ey1, ex0, ex1):
        if cache is None:
            return None
        return cache[..., ey0:ey1, ex0:ex1]

    def _conv2_gemm_chunk_into(
        self, chunk_index, prepared, destination,
        crop_y0, crop_y1, crop_x0, crop_x1,
    ):
        """Evaluate one conv2 output-channel chunk as bounded im2col + GEMM.

        The released conv2 checkpoint is already split into four 768-channel
        Comfy-managed modules. Keep exactly one such weight chunk resident at
        a time, while the explicit patch matrix is tiled over output H/W to
        cap transient VRAM. This avoids both the monolithic ~3072-channel
        Conv3d cast and the previous two-pass conv2 recomputation.
        """
        module = self.conv2.chunks[chunk_index]
        kt, kh, kw = (int(value) for value in module.kernel_size)
        st, sh, sw = (int(value) for value in module.stride)
        batch, in_channels, temporal, _, _ = prepared.shape
        out_t = (temporal - kt) // st + 1
        core_h = int(crop_y1 - crop_y0)
        core_w = int(crop_x1 - crop_x0)
        kernel_features = in_channels * kt * kh * kw

        if out_t < 1 or core_h < 1 or core_w < 1:
            raise RuntimeError("FlashVSR conv2 GEMM received an empty output tile.")

        budget_bytes = int(self.CONV2_IM2COL_BUDGET_MIB) * 1024 * 1024
        bytes_per_position = (
            batch * out_t * kernel_features * prepared.element_size()
        )
        positions = max(1, budget_bytes // max(1, bytes_per_position))
        cols = max(1, min(core_w, positions))
        rows = max(1, min(core_h, positions // cols))

        # The LQ projector is loaded with Comfy's manual-cast operations. Use
        # the same cast context as an ordinary Conv3d forward, but hold only
        # this one 768-channel weight chunk while its spatial slices execute.
        comfy.ops.run_every_op()
        with comfy.ops.CastBiasWeightContext(
            module, prepared, offloadable=True
        ) as (weight, bias):
            out_channels = int(weight.shape[0])
            weight_matrix = weight.reshape(
                out_channels, kernel_features
            ).t()

            for oy0 in range(crop_y0, crop_y1, rows):
                oy1 = min(crop_y1, oy0 + rows)
                for ox0 in range(crop_x0, crop_x1, cols):
                    ox1 = min(crop_x1, ox0 + cols)
                    source = prepared[
                        :, :, :,
                        oy0 * sh:(oy1 - 1) * sh + kh,
                        ox0 * sw:(ox1 - 1) * sw + kw,
                    ]
                    patches = (
                        source.unfold(2, kt, st)
                        .unfold(3, kh, sh)
                        .unfold(4, kw, sw)
                        .permute(0, 2, 3, 4, 1, 5, 6, 7)
                        .reshape(-1, kernel_features)
                    )
                    flat = (
                        torch.addmm(bias, patches, weight_matrix)
                        if bias is not None
                        else patches @ weight_matrix
                    )
                    block_h = oy1 - oy0
                    block_w = ox1 - ox0
                    block = flat.reshape(
                        batch, out_t, block_h, block_w, out_channels
                    ).permute(0, 4, 1, 2, 3)
                    destination[
                        :, :, :,
                        oy0 - crop_y0:oy1 - crop_y0,
                        ox0 - crop_x0:ox1 - crop_x0,
                    ].copy_(block)
                    del source, patches, flat, block

    def _project_tile_gemm(
        self, hidden, cache2_tile, crop_y0, crop_y1, crop_x0, crop_x1,
        profiler=None,
    ):
        """Project one spatial core with one conv2 pass and bounded GEMMs."""
        prepared = self.conv2.prepare_input(hidden, cache2_tile)
        module = self.conv2.chunks[0]
        kt = int(module.kernel_size[0])
        st = int(module.stride[0])
        tile_frames = (prepared.shape[2] - kt) // st + 1
        core_h = int(crop_y1 - crop_y0)
        core_w = int(crop_x1 - crop_x0)
        projected = prepared.new_empty((
            prepared.shape[0],
            self.conv2.out_channels,
            tile_frames,
            core_h,
            core_w,
        ))

        conv_marker = (
            profiler.profile_start(prepared)
            if profiler is not None else None
        )
        for chunk_index in range(self.conv2.chunk_count):
            channel_start = chunk_index * self.conv2.chunk_channels
            channel_end = channel_start + self.conv2.chunk_channels
            self._conv2_gemm_chunk_into(
                chunk_index,
                prepared,
                projected[:, channel_start:channel_end],
                crop_y0,
                crop_y1,
                crop_x0,
                crop_x1,
            )
        if profiler is not None:
            profiler.profile_end("lq_conv2", conv_marker)
        del prepared

        # Preserve the previous streamed RMS reduction: accumulate FP32
        # channel-norm contributions in the same 768-channel partitioning,
        # but reuse the already-computed conv2 output instead of recomputing it.
        norm_sq = None
        for chunk_index in range(self.conv2.chunk_count):
            channel_start = chunk_index * self.conv2.chunk_channels
            channel_end = channel_start + self.conv2.chunk_channels
            current_sq = projected[
                :, channel_start:channel_end
            ].float().square_().sum(dim=1, keepdim=True)
            if norm_sq is None:
                norm_sq = current_sq
            else:
                norm_sq.add_(current_sq)
        norm = norm_sq.sqrt_().to(dtype=projected.dtype)
        norm.clamp_min_(1.0e-12)
        del norm_sq

        tile_outputs = None
        for chunk_index in range(self.conv2.chunk_count):
            channel_start = chunk_index * self.conv2.chunk_channels
            channel_end = channel_start + self.conv2.chunk_channels
            chunk = projected[:, channel_start:channel_end]
            norm_marker = (
                profiler.profile_start(chunk)
                if profiler is not None else None
            )
            gamma = self.norm2.gamma[channel_start:channel_end].to(
                device=chunk.device, dtype=chunk.dtype
            )
            chunk.div_(norm)
            chunk.mul_(self.norm2.scale)
            chunk.mul_(gamma)
            F.silu(chunk, inplace=True)
            if profiler is not None:
                profiler.profile_end("lq_norm_act2", norm_marker)

            tile_tokens = rearrange(
                chunk, "b c f h w -> b (f h w) c"
            )
            linear_marker = (
                profiler.profile_start(tile_tokens)
                if profiler is not None else None
            )
            partials = [
                layer.forward_chunk(chunk_index, tile_tokens)
                for layer in self.linear_layers
            ]
            if profiler is not None:
                profiler.profile_end("lq_linear", linear_marker)
            if tile_outputs is None:
                tile_outputs = partials
            else:
                for output, partial in zip(tile_outputs, partials):
                    output.add_(partial)
            del chunk, gamma, tile_tokens, partials

        del projected, norm
        return tile_outputs, tile_frames

    def _run_spatial_tiles(self, x, initialize_only, profiler=None):
        batch, _, _, height, width = x.shape
        old_cache1 = self.cache["conv1"]
        old_cache2 = self.cache["conv2"]
        next_cache2 = None
        output_grids = None

        for (
            y0, y1, x0, x1,
            ey0, ey1, ex0, ex1,
            crop_y0, crop_x0,
        ) in self._spatial_tiles(height, width):
            core_h = y1 - y0
            core_w = x1 - x0
            crop_y1 = crop_y0 + core_h
            crop_x1 = crop_x0 + core_w

            tile = x[..., ey0:ey1, ex0:ex1]
            cache1_tile = self._cache_tile(
                old_cache1, ey0, ey1, ex0, ex1
            )
            conv1_marker = (
                profiler.profile_start(tile)
                if profiler is not None else None
            )
            hidden = self.conv1(tile, cache1_tile)
            if profiler is not None:
                profiler.profile_end("lq_conv1", conv1_marker)
            norm1_marker = (
                profiler.profile_start(hidden)
                if profiler is not None else None
            )
            hidden = self.act1(self.norm1(hidden))
            if profiler is not None:
                profiler.profile_end("lq_norm_act1", norm1_marker)

            current_tail = hidden[:, :, -CACHE_T:]
            if next_cache2 is None:
                next_cache2 = hidden.new_empty((
                    batch, hidden.shape[1], current_tail.shape[2],
                    height, width,
                ))
            next_cache2[:, :, :, y0:y1, x0:x1].copy_(
                current_tail[
                    :, :, :,
                    crop_y0:crop_y1,
                    crop_x0:crop_x1,
                ]
            )

            if initialize_only:
                del tile, hidden, current_tail
                continue

            cache2_tile = self._cache_tile(
                old_cache2, ey0, ey1, ex0, ex1
            )
            tile_outputs, tile_frames = self._project_tile_gemm(
                hidden,
                cache2_tile,
                crop_y0,
                crop_y1,
                crop_x0,
                crop_x1,
                profiler=profiler,
            )

            if output_grids is None:
                output_grids = [
                    value.new_empty((
                        batch, tile_frames, height, width, value.shape[-1],
                    ))
                    for value in tile_outputs
                ]
            for grid, value in zip(output_grids, tile_outputs):
                grid[:, :, y0:y1, x0:x1, :].copy_(
                    value.reshape(
                        batch, tile_frames, core_h, core_w, value.shape[-1],
                    )
                )
            del tile, hidden, current_tail, cache2_tile, tile_outputs

        # Do not update either temporal cache until all tiles have consumed
        # the old cache, otherwise halo overlap can observe current-clip state.
        cache_marker = (
            profiler.profile_start(x) if profiler is not None else None
        )
        self._store_causal_tail("conv1", x)
        if old_cache2 is not None and old_cache2.shape == next_cache2.shape:
            old_cache2.copy_(next_cache2)
            self.cache["conv2"] = old_cache2
        else:
            self.cache["conv2"] = next_cache2
        if profiler is not None:
            profiler.profile_end("lq_cache_update", cache_marker)

        if initialize_only:
            return None
        return [
            grid.reshape(batch, -1, grid.shape[-1])
            for grid in output_grids
        ]

    def stream_forward(self, video_clip: torch.Tensor, profiler=None):
        input_marker = (
            profiler.profile_start(video_clip)
            if profiler is not None else None
        )
        video_clip = video_clip.to(dtype=self.compute_dtype)
        if profiler is not None:
            profiler.profile_end("lq_input_cast", input_marker)

        pixel_marker = (
            profiler.profile_start(video_clip)
            if profiler is not None else None
        )
        initialize_only = self.clip_idx == 0
        if initialize_only:
            first = video_clip[:, :, :1].repeat(1, 1, 3, 1, 1)
            x = self.pixel_shuffle(torch.cat((first, video_clip), dim=2))
            del first
        else:
            x = self.pixel_shuffle(video_clip)
        if profiler is not None:
            profiler.profile_end("lq_pixel_unshuffle", pixel_marker)

        output = self._run_spatial_tiles(
            x, initialize_only=initialize_only, profiler=profiler
        )
        self.clip_idx += 1
        return output


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, operations, n_in, n_out, device=None, dtype=None):
        super().__init__()
        conv = lambda a, b, **kw: operations.Conv2d(a, b, 3, padding=1, device=device, dtype=dtype, **kw)
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = operations.Conv2d(n_in, n_out, 1, bias=False, device=device, dtype=dtype) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat((x, past), dim=1)) + self.skip(x))


class TGrow(nn.Module):
    def __init__(self, operations, channels, stride, device=None, dtype=None):
        super().__init__()
        self.stride = stride
        self.conv = operations.Conv2d(channels, channels * stride, 1, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        nt, channels, height, width = x.shape
        return self.conv(x).reshape(-1, channels, height, width)


class FusedTGrowConv(nn.Module):
    """Compose TGrow's 1x1 projection with its following 3x3 convolution."""

    def __init__(self, operations, n_in, n_out, stride,
                 device=None, dtype=None):
        super().__init__()
        self.stride = int(stride)
        self.n_out = int(n_out)
        self.fused = operations.Conv2d(
            n_in,
            n_out * self.stride,
            3,
            padding=1,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        nt, _, height, width = x.shape
        return self.fused(x).reshape(
            nt * self.stride,
            self.n_out,
            height,
            width,
        )


def _identity_conv(operations, channels, device=None, dtype=None):
    # It is loaded from the checkpoint. Keeping the Conv2d directly in the
    # Sequential preserves decoder.<index>.weight key compatibility.
    layer = operations.Conv2d(channels, channels, 3, padding=1, bias=False, device=device, dtype=dtype)
    layer.flashvsr_identity_layer = True
    return layer


class _TCDecoderProfiler:
    """Low-overhead CUDA-event profiler local to one decoder invocation."""

    def __init__(self, enabled, device, temporal_batch_size):
        self.device = torch.device(device)
        self.enabled = bool(
            enabled and self.device.type == "cuda" and torch.cuda.is_available()
        )
        self.events = {}
        self.wall_start = time.perf_counter()
        self.temporal_batch_size = int(temporal_batch_size)
        self.start_allocated = 0
        self.start_reserved = 0
        if self.enabled:
            with torch.cuda.device(self.device):
                self.start_allocated = torch.cuda.memory_allocated(self.device)
                self.start_reserved = torch.cuda.memory_reserved(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)

    def start(self):
        if not self.enabled:
            return None
        with torch.cuda.device(self.device):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return event

    def end(self, name, start):
        if start is None:
            return
        with torch.cuda.device(self.device):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
        self.events.setdefault(name, []).append((start, end))

    @staticmethod
    def block_name(block, current, base_height):
        scale = max(1, int(round(current.shape[-2] / max(1, base_height))))
        if isinstance(block, MemBlock):
            return f"tc_memblock_{scale}x"
        if isinstance(block, nn.Upsample):
            return f"tc_upsample_{scale}x"
        if isinstance(block, (TGrow, FusedTGrowConv)):
            return f"tc_tgrow_{scale}x"
        if isinstance(block, nn.Conv2d):
            return f"tc_conv_{scale}x"
        return f"tc_elementwise_{scale}x"

    def finish(self, latent_frames, output_frames):
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        wall_ms = (time.perf_counter() - self.wall_start) * 1000.0
        totals = {
            name: sum(start.elapsed_time(end) for start, end in records)
            for name, records in self.events.items()
        }
        counts = {name: len(records) for name, records in self.events.items()}
        print(
            "[FlashVSR TC profiler] "
            f"temporal_batch_size={self.temporal_batch_size}, "
            f"latent_frames={latent_frames}, output_frames={output_frames}."
        )
        print(f"[FlashVSR TC profiler]   wall_total: {wall_ms:.2f} ms")
        preferred = (
            "tc_decode_cuda_total",
            "tc_condition_transfer",
            "tc_pixel_unshuffle",
            "tc_latent_pack_normalize",
            "tc_state_update",
            "tc_crop_clamp",
            "tc_output_staging",
            "tc_output_copy",
        )
        ordered = list(preferred) + sorted(
            name for name in totals if name not in preferred
        )
        for name in ordered:
            if name not in totals:
                continue
            total = totals[name]
            count = counts[name]
            print(
                f"[FlashVSR TC profiler]   {name}: {total:.2f} ms "
                f"({count} calls, {total / count:.2f} ms/call)"
            )
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        print(
            "[FlashVSR TC profiler]   memory: "
            f"start_allocated={self.start_allocated / (1024 ** 2):.0f} MiB, "
            f"peak_allocated={peak_allocated / (1024 ** 2):.0f} MiB, "
            f"start_reserved={self.start_reserved / (1024 ** 2):.0f} MiB, "
            f"peak_reserved={peak_reserved / (1024 ** 2):.0f} MiB."
        )
        if output_frames:
            print(
                "[FlashVSR TC profiler]   throughput: "
                f"{wall_ms / output_frames:.2f} ms/output frame."
            )


class TCDecoder(ManagedComponent):
    image_channels = 3

    def __init__(self, operations, compute_dtype, device=None,
                 weight_dtype=None, fuse_tgrow=False,
                 channels_last=False, compile_memblocks=False):
        super().__init__(compute_dtype)
        channels = [512, 256, 128, 128]
        latent_channels = 16 + 768
        conv = lambda a, b, **kw: operations.Conv2d(a, b, 3, padding=1, device=device, dtype=weight_dtype, **kw)

        base = nn.Sequential(
            Clamp(), conv(latent_channels, channels[0]), nn.ReLU(inplace=True),
            MemBlock(operations, channels[0], channels[0], device, weight_dtype),
            MemBlock(operations, channels[0], channels[0], device, weight_dtype),
            MemBlock(operations, channels[0], channels[0], device, weight_dtype),
            nn.Upsample(scale_factor=2), TGrow(operations, channels[0], 1, device, weight_dtype),
            conv(channels[0], channels[1], bias=False),
            MemBlock(operations, channels[1], channels[1], device, weight_dtype),
            MemBlock(operations, channels[1], channels[1], device, weight_dtype),
            MemBlock(operations, channels[1], channels[1], device, weight_dtype),
            nn.Upsample(scale_factor=2), TGrow(operations, channels[1], 2, device, weight_dtype),
            conv(channels[1], channels[2], bias=False),
            MemBlock(operations, channels[2], channels[2], device, weight_dtype),
            MemBlock(operations, channels[2], channels[2], device, weight_dtype),
            MemBlock(operations, channels[2], channels[2], device, weight_dtype),
            nn.Upsample(scale_factor=2), TGrow(operations, channels[2], 2, device, weight_dtype),
            conv(channels[2], channels[3], bias=False),
            nn.ReLU(inplace=True), conv(channels[3], self.image_channels),
        )
        self.decoder = self._deepen(base, operations, device, weight_dtype)
        self.fuse_tgrow = bool(fuse_tgrow)
        self.use_channels_last = bool(channels_last)
        self.compile_memblocks = bool(compile_memblocks)
        self._compiled_memblocks = {}
        self._compile_disabled_reason = None
        self._compile_reported = False
        if self.fuse_tgrow:
            self._fuse_tgrow_layers(
                operations, device=device, dtype=weight_dtype
            )
        self.pixel_shuffle = PixelUnshuffle3D(4, 8, 8)
        self.frames_to_trim = 3
        self.clean_mem()

    @staticmethod
    def _deepen(base, operations, device, dtype):
        layers = []
        for block in base:
            layers.append(block)
            if isinstance(block, nn.ReLU):
                channels = None
                previous = layers[-2] if len(layers) >= 2 else None
                if isinstance(previous, nn.Conv2d):
                    channels = previous.out_channels
                elif isinstance(previous, MemBlock):
                    channels = previous.conv[-1].out_channels
                if channels is not None:
                    layers.extend((_identity_conv(operations, channels, device, dtype), nn.ReLU(inplace=True)))
        return nn.Sequential(*layers)

    def _fuse_tgrow_layers(self, operations, device=None, dtype=None):
        """Replace each linear TGrow->Conv2d pair without shifting keys."""
        for index in range(len(self.decoder) - 1):
            grow = self.decoder[index]
            following = self.decoder[index + 1]
            if not isinstance(grow, TGrow) or not isinstance(
                following, nn.Conv2d
            ):
                continue
            fused = FusedTGrowConv(
                operations,
                grow.conv.in_channels,
                following.out_channels,
                grow.stride,
                device=device,
                dtype=dtype,
            )
            self.decoder[index] = fused
            # Retaining the Sequential index keeps every later released
            # checkpoint key stable. patch_state_dict consumes the following
            # Conv2d weight while constructing decoder.<index>.fused.weight.
            self.decoder[index + 1] = nn.Identity()

    def clean_mem(self):
        self.mem = [None] * len(self.decoder)

    def optimize_memory_format(self):
        """Optionally keep Conv2d weights in channels-last format."""
        if self.use_channels_last:
            self.to(memory_format=torch.channels_last)

    def _channels_last(self, x):
        if (
            self.use_channels_last
            and x.ndim == 4
            and not x.is_contiguous(
                memory_format=torch.channels_last
            )
        ):
            return x.contiguous(memory_format=torch.channels_last)
        return x

    def _clone_state(self, current):
        memory_format = (
            torch.channels_last
            if self.use_channels_last
            else torch.contiguous_format
        )
        return current.detach().clone(memory_format=memory_format)

    @staticmethod
    def _has_exact_storage(current):
        """True when retaining current cannot pin a larger sibling batch."""
        try:
            return (
                current.storage_offset() == 0
                and current.untyped_storage().nbytes()
                == current.numel() * current.element_size()
            )
        except (AttributeError, RuntimeError):
            return False

    def _retain_state(self, index, current):
        """Retain one causal frame without retaining a temporal parent."""
        if self._has_exact_storage(current):
            self.mem[index] = current.detach()
            return

        stored = self.mem[index]
        if (
            stored is not None
            and stored.shape == current.shape
            and stored.device == current.device
            and stored.dtype == current.dtype
        ):
            stored.copy_(current)
            self.mem[index] = stored
        else:
            self.mem[index] = self._clone_state(current)

    def _call_memblock(self, block, current, past):
        """Run an optional static-shape Inductor wrapper with eager fallback.

        Keep compiled wrappers outside the registered module tree so ComfyUI's
        CoreModelPatcher continues to own the original convolution parameters
        and their Dynamic VRAM lifetime.
        """
        if not self.compile_memblocks or self._compile_disabled_reason:
            return block(current, past)
        compiler = getattr(torch, "compile", None)
        if not callable(compiler):
            self._compile_disabled_reason = "torch.compile is unavailable"
            print(
                "[FlashVSR] TCDecoder MemBlock compilation unavailable: "
                f"{self._compile_disabled_reason}; using eager execution."
            )
            return block(current, past)
        key = id(block)
        compiled = self._compiled_memblocks.get(key)
        if compiled is None:
            try:
                compiled = compiler(
                    block,
                    backend="inductor",
                    dynamic=False,
                    fullgraph=False,
                )
                self._compiled_memblocks[key] = compiled
                if not self._compile_reported:
                    print(
                        "[FlashVSR] TCDecoder MemBlock torch.compile enabled; "
                        "the first invocation of each static block may be "
                        "slower while Inductor compiles it."
                    )
                    self._compile_reported = True
            except Exception as error:
                self._compile_disabled_reason = str(error)
                self._compiled_memblocks.clear()
                print(
                    "[FlashVSR] TCDecoder MemBlock compilation setup failed: "
                    f"{error}; using eager execution."
                )
                return block(current, past)
        try:
            return compiled(current, past)
        except Exception as error:
            self._compile_disabled_reason = str(error)
            self._compiled_memblocks.clear()
            print(
                "[FlashVSR] TCDecoder compiled MemBlock failed: "
                f"{error}; using eager execution for this and later blocks."
            )
            return block(current, past)

    def _run_sequential(self, input_frame, profiler=None):
        """Depth-first execution with one high-resolution branch live."""
        pending = deque(((self._channels_last(input_frame), 0),))
        base_height = input_frame.shape[-2]

        while pending:
            current, index = pending.popleft()
            if index == len(self.decoder):
                # Yield immediately so decode_video can stage/offload this
                # frame before the next high-resolution branch is evaluated.
                yield current
                continue

            block = self.decoder[index]
            stage_name = (
                profiler.block_name(block, current, base_height)
                if profiler is not None else None
            )
            marker = profiler.start() if profiler is not None else None
            if isinstance(block, MemBlock):
                stored = self.mem[index]
                past = torch.zeros_like(current) if stored is None else stored
                updated = self._call_memblock(block, current, past)
                if profiler is not None:
                    profiler.end(stage_name, marker)
                # Fused TGrow emits views into one multi-frame allocation.
                # Copy those views into a compact state instead of pinning all
                # sibling branches; transfer ownership for ordinary outputs.
                state_marker = profiler.start() if profiler is not None else None
                self._retain_state(index, current)
                if profiler is not None:
                    profiler.end("tc_state_update", state_marker)
                pending.appendleft((self._channels_last(updated), index + 1))
                del current, past, stored, updated
                continue

            current = self._channels_last(block(current))
            if profiler is not None:
                profiler.end(stage_name, marker)
            if isinstance(block, (TGrow, FusedTGrowConv)):
                n, channels, height, width = current.shape
                stride = block.stride
                grown = current.reshape(
                    n // stride, stride, channels, height, width
                )
                # appendleft in reverse preserves temporal order while fully
                # completing one branch before the next branch is evaluated.
                for branch in range(stride - 1, -1, -1):
                    pending.appendleft((grown[:, branch], index + 1))
                del current, grown
            else:
                pending.appendleft((current, index + 1))

    def _run_temporal_batch(self, inputs, profiler=None):
        """Execute a bounded NTCHW group while preserving causal MemBlocks."""
        n, temporal, channels, height, width = inputs.shape
        current = self._channels_last(
            inputs.reshape(n * temporal, channels, height, width)
        )
        base_height = height

        for index, block in enumerate(self.decoder):
            stage_name = (
                profiler.block_name(block, current, base_height)
                if profiler is not None else None
            )
            marker = profiler.start() if profiler is not None else None
            if isinstance(block, MemBlock):
                _, channels, height, width = current.shape
                current_nt = current.reshape(
                    n, temporal, channels, height, width
                )
                stored = self.mem[index]
                if temporal == 1:
                    if stored is None:
                        past = torch.zeros_like(current)
                    else:
                        past = stored
                    updated = self._call_memblock(block, current, past)
                    if profiler is not None:
                        profiler.end(stage_name, marker)
                    state_marker = profiler.start() if profiler is not None else None
                    self._retain_state(index, current)
                    if profiler is not None:
                        profiler.end("tc_state_update", state_marker)
                else:
                    past = torch.empty_like(current)
                    past_nt = past.reshape(
                        n, temporal, channels, height, width
                    )
                    if stored is None:
                        past_nt[:, 0].zero_()
                    else:
                        past_nt[:, 0].copy_(stored)
                    past_nt[:, 1:].copy_(current_nt[:, :-1])
                    updated = self._call_memblock(block, current, past)
                    if profiler is not None:
                        profiler.end(stage_name, marker)
                    # A view of the last item would retain the complete batch.
                    # Clone only this one bounded state at batch boundaries.
                    state_marker = (
                        profiler.start() if profiler is not None else None
                    )
                    self.mem[index] = self._clone_state(current_nt[:, -1])
                    if profiler is not None:
                        profiler.end("tc_state_update", state_marker)
                current = self._channels_last(updated)
                del updated, past, stored
                continue

            current = block(current)
            current = self._channels_last(current)
            if profiler is not None:
                profiler.end(stage_name, marker)
            if isinstance(block, (TGrow, FusedTGrowConv)):
                temporal *= block.stride

        _, channels, height, width = current.shape
        return current.reshape(n, temporal, channels, height, width)

    def patch_state_dict(self, state_dict):
        """Accept the pre-pruning TGrow layout used by some released packs."""
        for index, layer in enumerate(self.decoder):
            key = f"decoder.{index}.conv.weight"
            if isinstance(layer, TGrow):
                if (
                    key in state_dict
                    and state_dict[key].shape[0] > layer.conv.weight.shape[0]
                ):
                    state_dict[key] = state_dict[key][
                        -layer.conv.weight.shape[0]:
                    ]
                continue
            if not isinstance(layer, FusedTGrowConv):
                continue

            following_key = f"decoder.{index + 1}.weight"
            if key not in state_dict or following_key not in state_dict:
                continue
            grow_weight = state_dict[key]
            following_weight = state_dict[following_key]
            expected_grow_channels = (
                layer.fused.in_channels * layer.stride
            )
            if grow_weight.shape[0] > expected_grow_channels:
                grow_weight = grow_weight[-expected_grow_channels:]
            grow_matrix = grow_weight.reshape(
                layer.stride,
                expected_grow_channels // layer.stride,
                layer.fused.in_channels,
            ).float()
            # For each temporal branch: following_3x3 @ grow_1x1. Compose in
            # FP32 once on CPU, then store in the decoder's native weight dtype.
            fused_weight = torch.einsum(
                "omhw,smi->soihw",
                following_weight.float(),
                grow_matrix,
            ).reshape_as(layer.fused.weight).to(
                device="cpu", dtype=grow_weight.dtype
            )
            state_dict[f"decoder.{index}.fused.weight"] = fused_weight
            del state_dict[key], state_dict[following_key]
        return state_dict

    @torch.inference_mode()
    def decode_video(
        self,
        latents_bcthw,
        condition_bcfhw,
        *,
        compute_device=None,
        compute_dtype=None,
        latent_mean=None,
        latent_std=None,
        latent_scale_factor=1.0,
        output_device=None,
        output_dtype=None,
        output_chunk_size=4,
        temporal_batch_size=1,
        frame_start=0,
        frame_count=None,
        output_height=None,
        output_width=None,
        clamp_output=False,
        profile_cuda_events=False,
    ):
        condition_provider = callable(
            getattr(condition_bcfhw, "condition_frames", None)
        )
        if latents_bcthw.ndim != 5:
            raise ValueError("TCDecoder expects a five-dimensional latent.")
        if not condition_provider and condition_bcfhw.ndim != 5:
            raise ValueError(
                "TCDecoder expects either a five-dimensional conditioning "
                "tensor or a FlashVSR lazy conditioning provider."
            )
        n, _, timesteps, _, _ = latents_bcthw.shape
        required_condition_frames = 1 + max(0, timesteps - 1) * 4
        available_condition_frames = (
            int(condition_bcfhw.generated_frames)
            if condition_provider
            else int(condition_bcfhw.shape[2])
        )
        if available_condition_frames < required_condition_frames:
            raise RuntimeError(
                "TCDecoder conditioning video is too short: "
                f"{available_condition_frames} < "
                f"{required_condition_frames}."
            )

        compute_device = (
            latents_bcthw.device
            if compute_device is None
            else torch.device(compute_device)
        )
        compute_dtype = (
            self.compute_dtype if compute_dtype is None else compute_dtype
        )
        normalized_mean = (
            None
            if latent_mean is None
            else latent_mean.to(
                device=compute_device, dtype=compute_dtype
            )
        )
        normalized_std = (
            None
            if latent_std is None
            else latent_std.to(
                device=compute_device, dtype=compute_dtype
            )
        )

        normalized_scale = (
            None
            if normalized_std is None
            else float(latent_scale_factor) / normalized_std
        )
        if normalized_mean is not None:
            normalized_mean = normalized_mean[:, :, 0]
        if normalized_scale is not None:
            normalized_scale = normalized_scale[:, :, 0]

        output = []
        trim = self.mem[-8] is None
        selected_output = None
        selected_index = 0
        staged_output = None
        staged_count = 0
        output_chunk_size = max(1, int(output_chunk_size))
        temporal_batch_size = max(
            1, min(int(temporal_batch_size), timesteps)
        )
        profiler = _TCDecoderProfiler(
            profile_cuda_events, compute_device, temporal_batch_size
        )
        total_marker = profiler.start()
        produced_frames = 0
        raw_start = (self.frames_to_trim if trim else 0) + max(
            0, int(frame_start)
        )
        raw_stop = (
            raw_start + max(0, int(frame_count))
            if frame_count is not None
            else None
        )
        finished = False

        def flush_staged_output():
            nonlocal selected_index, staged_count
            if staged_count == 0:
                return
            marker = profiler.start()
            selected_output[
                :, selected_index:selected_index + staged_count
            ].copy_(staged_output[:, :staged_count])
            profiler.end("tc_output_copy", marker)
            selected_index += staged_count
            staged_count = 0

        condition_channels = (
            (3 if condition_provider else condition_bcfhw.shape[1])
            * self.pixel_shuffle.temporal
            * self.pixel_shuffle.height
            * self.pixel_shuffle.width
        )
        latent_channels = latents_bcthw.shape[1]
        latent_height, latent_width = latents_bcthw.shape[-2:]
        input_buffer = torch.empty(
            (
                n,
                temporal_batch_size,
                condition_channels + latent_channels,
                latent_height,
                latent_width,
            ),
            device=compute_device,
            dtype=compute_dtype,
        )

        for batch_start in range(0, timesteps, temporal_batch_size):
            batch_count = min(
                temporal_batch_size, timesteps - batch_start
            )
            for offset in range(batch_count):
                timestep = batch_start + offset
                if timestep == 0:
                    clip_start = 0
                    clip_count = 1
                else:
                    clip_start = 1 + (timestep - 1) * 4
                    clip_count = 4
                marker = profiler.start()
                if condition_provider:
                    condition_gpu = condition_bcfhw.condition_frames(
                        clip_start,
                        clip_count,
                        device=compute_device,
                        dtype=compute_dtype,
                    )
                else:
                    condition_clip = condition_bcfhw[
                        :, :, clip_start:clip_start + clip_count
                    ]
                    condition_gpu = condition_clip.to(
                        device=compute_device, dtype=compute_dtype
                    )
                profiler.end("tc_condition_transfer", marker)
                marker = profiler.start()
                condition_t = self.pixel_shuffle(condition_gpu)
                profiler.end("tc_pixel_unshuffle", marker)
                if condition_t.shape[2] != 1:
                    raise RuntimeError(
                        "TCDecoder conditioning chunk did not produce "
                        "exactly one latent timestep."
                    )
                current_input = input_buffer[:, offset]
                marker = profiler.start()
                current_input[:, :condition_channels].copy_(
                    condition_t[:, :, 0]
                )
                latent_input = current_input[:, condition_channels:]
                latent_input.copy_(
                    latents_bcthw[:, :, timestep]
                )
                if (
                    normalized_mean is not None
                    and normalized_scale is not None
                ):
                    latent_input.sub_(normalized_mean).mul_(
                        normalized_scale
                    )
                profiler.end("tc_latent_pack_normalize", marker)
                del condition_gpu, condition_t, current_input, latent_input

            if temporal_batch_size == 1:
                decoded = None
                decoded_frames = self._run_sequential(
                    input_buffer[:, 0], profiler
                )
            else:
                decoded = self._run_temporal_batch(
                    input_buffer[:, :batch_count], profiler
                )
                decoded_frames = (
                    decoded[:, output_index]
                    for output_index in range(decoded.shape[1])
                )
            for current in decoded_frames:
                if raw_stop is None:
                    output.append(current)
                elif raw_start <= produced_frames < raw_stop:
                    frame = current
                    marker = profiler.start()
                    if output_height is not None:
                        frame = frame[:, :, :int(output_height)]
                    if output_width is not None:
                        frame = frame[:, :, :, :int(output_width)]
                    if clamp_output:
                        frame.clamp_(0.0, 1.0)
                    profiler.end("tc_crop_clamp", marker)
                    target_device = (
                        frame.device
                        if output_device is None
                        else output_device
                    )
                    target_dtype = (
                        frame.dtype
                        if output_dtype is None
                        else output_dtype
                    )
                    if selected_output is None:
                        selected_output = torch.empty(
                            (
                                n,
                                int(frame_count),
                                frame.shape[1],
                                frame.shape[2],
                                frame.shape[3],
                            ),
                            device=target_device,
                            dtype=target_dtype,
                        )
                        staged_output = torch.empty(
                            (
                                n,
                                min(
                                    output_chunk_size,
                                    int(frame_count),
                                ),
                                frame.shape[1],
                                frame.shape[2],
                                frame.shape[3],
                            ),
                            device=frame.device,
                            dtype=frame.dtype,
                        )
                    marker = profiler.start()
                    staged_output[:, staged_count].copy_(frame)
                    profiler.end("tc_output_staging", marker)
                    staged_count += 1
                    if staged_count == staged_output.shape[1]:
                        flush_staged_output()
                    del frame
                produced_frames += 1
                if raw_stop is not None and produced_frames >= raw_stop:
                    flush_staged_output()
                    finished = True
                    break
            del current, decoded_frames, decoded
            if finished:
                break

        if raw_stop is not None:
            if selected_output is None or selected_index != int(frame_count):
                raise RuntimeError(
                    "TCDecoder produced an incomplete selected frame range: "
                    f"{selected_index} of {frame_count} frames."
                )
            profiler.end("tc_decode_cuda_total", total_marker)
            profiler.finish(timesteps, selected_index)
            return selected_output
        frames = torch.stack(output, dim=1)
        result = frames[:, self.frames_to_trim:] if trim else frames
        profiler.end("tc_decode_cuda_total", total_marker)
        profiler.finish(timesteps, result.shape[1])
        return result


def _load_managed(
    path,
    component_class,
    dtype_name: str,
    prefix: str = "",
    store_compute_dtype: bool = False,
    force_manual_cast: bool = False,
    component_kwargs=None,
) -> ComponentHandle:
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if prefix:
        sd = {key.removeprefix(prefix): value for key, value in sd.items()}
    checkpoint_dtype = _state_dtype(sd)
    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()
    compute_dtype = _component_dtype(dtype_name, load_device)
    weight_dtype = compute_dtype if store_compute_dtype else checkpoint_dtype

    # TCDecoder and the LQ projector reuse large convolution weights throughout
    # a run. Keeping an FP32 checkpoint in ComfyUI's manual-cast path would
    # recast those weights repeatedly. Convert each floating tensor once on CPU
    # so CoreModelPatcher dynamically loads/offloads native compute-dtype weights.
    # Replace entries incrementally instead of constructing a second complete
    # state-dict mapping and retaining all FP32 tensors until conversion ends.
    if store_compute_dtype and checkpoint_dtype != weight_dtype:
        for key in list(sd):
            value = sd[key]
            if torch.is_tensor(value) and value.is_floating_point():
                sd[key] = value.to(device="cpu", dtype=weight_dtype)

    # A managed component may be offloaded between executions while the
    # surrounding diffusion-model clone remains resident. Manual-cast ops are
    # a safe fallback in that state: resident weights are used directly, while
    # offloaded weights are cast/moved through ComfyUI instead of being passed
    # as CPU tensors to a CUDA kernel.
    operations = (
        comfy.ops.manual_cast
        if force_manual_cast
        else comfy.ops.pick_operations(
            weight_dtype, compute_dtype, load_device=load_device
        )
    )
    component_kwargs = dict(component_kwargs or {})
    model = component_class(
        operations=operations,
        compute_dtype=compute_dtype,
        device=offload_device,
        weight_dtype=weight_dtype,
        **component_kwargs,
    ).eval()
    if hasattr(model, "patch_state_dict"):
        sd = model.patch_state_dict(sd)
    model_management.archive_model_dtypes(model)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=patcher.is_dynamic())
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {path}: missing={missing}, unexpected={unexpected}")
    if hasattr(model, "optimize_memory_format"):
        model.optimize_memory_format()
    return ComponentHandle(model=model, patcher=patcher, compute_dtype=compute_dtype)


def load_lq_projector(path: str, dtype_name: str = "auto") -> ComponentHandle:
    return _load_managed(
        path,
        LQProjector,
        dtype_name,
        prefix="LQ_proj_in.",
        store_compute_dtype=True,
        force_manual_cast=True,
    )


def load_tcdecoder(
    path: str,
    dtype_name: str = "auto",
    fuse_tgrow: bool = False,
    channels_last: bool = False,
    compile_memblocks: bool = False,
) -> ComponentHandle:
    return _load_managed(
        path,
        TCDecoder,
        dtype_name,
        store_compute_dtype=True,
        component_kwargs={
            "fuse_tgrow": bool(fuse_tgrow),
            "channels_last": bool(channels_last),
            "compile_memblocks": bool(compile_memblocks),
        },
    )
