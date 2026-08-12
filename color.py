"""Bounded wavelet and AdaIN color correction for FlashVSR output.

The decomposition follows the Apache-2.0 FlashVSR reference implementation,
adapted to ComfyUI IMAGE tensors and bounded frame chunks.
"""

from __future__ import annotations

from functools import lru_cache
import math

import torch
import torch.nn.functional as F


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
    difference = style - content
    downsample = max(1, int(downsample))
    if downsample > 1 and min(difference.shape[-2:]) > 1:
        height, width = difference.shape[-2:]
        working_height = max(1, (height + downsample - 1) // downsample)
        working_width = max(1, (width + downsample - 1) // downsample)
        difference = F.interpolate(
            difference,
            size=(working_height, working_width),
            mode="area",
        )
        removed_scales = max(0, int(round(math.log2(downsample))))
        difference = _wavelet_lowpass(
            difference, max(1, int(levels) - removed_scales)
        )
        difference = F.interpolate(
            difference,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    else:
        difference = _wavelet_lowpass(difference, int(levels))
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
) -> torch.Tensor:
    """Correct a cropped IMAGE batch without retaining every output chunk."""
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
    output = torch.empty_like(images)
    for offset in range(0, frame_count, chunk_size):
        end = min(offset + chunk_size, frame_count)
        content_chunk = images[offset:end, ..., :3]
        # Work in float32 because CPU float16 grouped convolution is not
        # consistently implemented across supported PyTorch versions.
        work_dtype = (
            content_chunk.dtype
            if content_chunk.is_cuda and content_chunk.dtype in (
                torch.float16, torch.bfloat16, torch.float32
            )
            else torch.float32
        )
        content_nchw = content_chunk.movedim(-1, 1).to(dtype=work_dtype)
        style_nchw = (
            style[offset:end, ..., :3]
            .movedim(-1, 1)
            .to(device=content_nchw.device, dtype=work_dtype)
            .add(1.0)
            .mul(0.5)
        )
        if method.startswith("wavelet_"):
            corrected = wavelet_reconstruct(
                content_nchw,
                style_nchw,
                levels=levels,
                downsample=(
                    4 if method == "wavelet_quarter_res" else 1
                ),
            )
        else:
            corrected = adain_reconstruct(content_nchw, style_nchw)
        corrected.clamp_(0.0, 1.0)
        corrected = corrected.movedim(1, -1).to(dtype=images.dtype)
        output[offset:end, ..., :3].copy_(corrected)
        if images.shape[-1] > 3:
            output[offset:end, ..., 3:].copy_(images[offset:end, ..., 3:])
    return output
