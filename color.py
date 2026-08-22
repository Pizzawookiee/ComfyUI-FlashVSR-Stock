"""Bounded wavelet and AdaIN color correction for FlashVSR output.

The decomposition follows the Apache-2.0 FlashVSR reference implementation,
adapted to ComfyUI IMAGE tensors and bounded frame chunks.
"""

from __future__ import annotations

from functools import lru_cache
import math
import time

import torch
import torch.nn.functional as F


class _ColorProfiler:
    """Optional wall/CUDA-event profiler for bounded color correction."""

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.cuda = bool(
            self.enabled
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self.wall_start = time.perf_counter()
        self.events = {}
        self.cpu = {}

    def start(self):
        if not self.enabled:
            return None
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        return event

    def end(self, name, marker):
        if marker is None:
            return
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.events.setdefault(name, []).append((marker, event))
        else:
            elapsed = (time.perf_counter() - marker) * 1000.0
            total, count = self.cpu.get(name, (0.0, 0))
            self.cpu[name] = (total + elapsed, count + 1)

    def finish(self, method, frames, chunk_size, inplace):
        if not self.enabled:
            return
        if self.cuda:
            torch.cuda.synchronize(self.device)
            totals = {
                name: (
                    sum(start.elapsed_time(end) for start, end in records),
                    len(records),
                )
                for name, records in self.events.items()
            }
        else:
            totals = self.cpu
        wall_ms = (time.perf_counter() - self.wall_start) * 1000.0
        print(
            "[FlashVSR Postprocess profiler] "
            f"method={method}, device={self.device}, frames={frames}, "
            f"chunk_size={chunk_size}, inplace={bool(inplace)}."
        )
        print(
            f"[FlashVSR Postprocess profiler]   wall_total: {wall_ms:.2f} ms"
        )
        for name in (
            "input_transfer", "downsample", "lowpass", "upsample_correct",
            "adain", "clamp", "output_transfer",
        ):
            if name not in totals:
                continue
            total, count = totals[name]
            print(
                f"[FlashVSR Postprocess profiler]   {name}: {total:.2f} ms "
                f"({count} calls, {total / count:.2f} ms/call)"
            )


@lru_cache(maxsize=16)
def _wavelet_weight(
    device: str, dtype: torch.dtype, channels: int
) -> torch.Tensor:
    kernel = torch.tensor(
        (
            (0.0625, 0.125, 0.0625),
            (0.125, 0.25, 0.125),
            (0.0625, 0.125, 0.0625),
        ),
        dtype=dtype,
        device=torch.device(device),
    )
    return kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    weight = _wavelet_weight(str(x.device), x.dtype, x.shape[1])
    padded = F.pad(
        x, (radius, radius, radius, radius), mode="replicate"
    )
    return F.conv2d(
        padded, weight, dilation=radius, groups=x.shape[1]
    )


def _wavelet_lowpass(x: torch.Tensor, levels: int) -> torch.Tensor:
    for level in range(levels):
        x = _wavelet_blur(x, 2**level)
    return x


def wavelet_reconstruct(
    content: torch.Tensor,
    style: torch.Tensor,
    levels: int = 5,
    downsample: int = 4,
    profiler=None,
) -> torch.Tensor:
    """Use generated detail and reference low-frequency color.

    The reference implementation computes
    ``content - lowpass(content) + lowpass(style)``. Because the complete
    dilated blur stack is linear, this is equivalently
    ``content + lowpass(style - content)`` and needs half as many convolutions.
    The color difference can be evaluated at reduced resolution because only
    its low-frequency component is retained. ``downsample=1`` preserves the
    previous exact five-pass path.
    """
    downsample = max(1, int(downsample))
    if downsample > 1 and min(content.shape[-2:]) > 1:
        height, width = content.shape[-2:]
        working_height = max(1, (height + downsample - 1) // downsample)
        working_width = max(1, (width + downsample - 1) // downsample)
        marker = profiler.start() if profiler is not None else None
        # Area resampling is linear. Downsample first and subtract second so
        # quarter-resolution mode never materializes a full-resolution
        # style-content difference tensor.
        content_small = F.interpolate(
            content,
            size=(working_height, working_width),
            mode="area",
        )
        style_small = F.interpolate(
            style,
            size=(working_height, working_width),
            mode="area",
        )
        difference = style_small.sub(content_small)
        if profiler is not None:
            profiler.end("downsample", marker)
        removed_scales = max(0, int(round(math.log2(downsample))))
        marker = profiler.start() if profiler is not None else None
        difference = _wavelet_lowpass(
            difference, max(1, int(levels) - removed_scales)
        )
        if profiler is not None:
            profiler.end("lowpass", marker)
        marker = profiler.start() if profiler is not None else None
        correction = F.interpolate(
            difference,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        result = content + correction
        if profiler is not None:
            profiler.end("upsample_correct", marker)
        return result
    else:
        difference = style - content
        marker = profiler.start() if profiler is not None else None
        difference = _wavelet_lowpass(difference, int(levels))
        if profiler is not None:
            profiler.end("lowpass", marker)
    return content + difference


def adain_reconstruct(
    content: torch.Tensor,
    style: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Match per-frame, per-channel mean and variance to the reference."""
    content_var, content_mean = torch.var_mean(
        content, dim=(2, 3), unbiased=False, keepdim=True
    )
    style_var, style_mean = torch.var_mean(
        style, dim=(2, 3), unbiased=False, keepdim=True
    )
    normalized = (content - content_mean) * torch.rsqrt(
        content_var + eps
    )
    return normalized * torch.sqrt(style_var + eps) + style_mean


def apply_color_correction(
    images: torch.Tensor,
    video,
    method: str,
    chunk_size: int = 4,
    levels: int = 5,
    compute_device: str = "auto",
    inplace: bool = True,
    profile_stages: bool = False,
) -> torch.Tensor:
    """Correct a cropped IMAGE batch with bounded CPU/GPU workspaces."""
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("Color correction expects IMAGE [F,H,W,C].")
    if method not in (
        "wavelet_quarter_res", "wavelet_full_res", "adain"
    ):
        raise ValueError(f"Unknown color correction method: {method}")

    frame_count, height, width, _ = images.shape
    start = video.crop_start
    style = video.tensor[
        0, :, start:start + frame_count, :height, :width
    ].permute(1, 2, 3, 0)
    if style.shape[0] != frame_count:
        raise RuntimeError(
            "Conditioning video is shorter than the decoded output selected "
            "for color correction."
        )

    chunk_size = max(1, int(chunk_size))
    requested = str(compute_device).lower()
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unknown color correction device: {compute_device}")
    if requested == "cpu":
        work_device = torch.device("cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA postprocessing was requested but unavailable.")
        work_device = torch.device("cuda", torch.cuda.current_device())
    elif images.is_cuda:
        work_device = images.device
    elif torch.cuda.is_available():
        work_device = torch.device("cuda", torch.cuda.current_device())
    else:
        work_device = torch.device("cpu")

    output = images if bool(inplace) else torch.empty_like(images)
    profiler = _ColorProfiler(profile_stages, work_device)
    for offset in range(0, frame_count, chunk_size):
        end = min(offset + chunk_size, frame_count)
        content_chunk = images[offset:end, ..., :3]
        # Work in float32 because CPU float16 grouped convolution is not
        # consistently implemented across supported PyTorch versions.
        work_dtype = (
            torch.float16 if work_device.type == "cuda" else torch.float32
        )
        marker = profiler.start()
        content_nchw = content_chunk.movedim(-1, 1).to(
            device=work_device, dtype=work_dtype
        )
        style_nchw = (
            style[offset:end, ..., :3]
            .movedim(-1, 1)
            .to(device=work_device, dtype=work_dtype)
            .add(1.0)
            .mul(0.5)
        )
        profiler.end("input_transfer", marker)
        if method.startswith("wavelet_"):
            corrected = wavelet_reconstruct(
                content_nchw,
                style_nchw,
                levels=levels,
                downsample=(
                    4 if method == "wavelet_quarter_res" else 1
                ),
                profiler=profiler,
            )
        else:
            marker = profiler.start()
            corrected = adain_reconstruct(content_nchw, style_nchw)
            profiler.end("adain", marker)
        marker = profiler.start()
        corrected.clamp_(0.0, 1.0)
        profiler.end("clamp", marker)
        marker = profiler.start()
        corrected = corrected.movedim(1, -1).to(
            device=output.device, dtype=images.dtype
        )
        output[offset:end, ..., :3].copy_(corrected)
        if images.shape[-1] > 3:
            output[offset:end, ..., 3:].copy_(images[offset:end, ..., 3:])
        profiler.end("output_transfer", marker)
        del content_nchw, style_nchw, corrected
    profiler.finish(method, frame_count, chunk_size, inplace)
    return output
