If you OOM on https://github.com/naxci1/ComfyUI-FlashVSR_Stable or https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast then try this, it works great on RTX 4050 6GB VRAM 16GB RAM.

# ComfyUI FlashVSR — Stock Wan

FlashVSR v1.1 video super-resolution built around ComfyUI's stock Wan model,
model patching, sampling, VAE, safetensors loading, dynamic VRAM management,
and attention-backend system.

This project is intended for users who want FlashVSR as a composable ComfyUI
workflow rather than an all-in-one pipeline. The FlashVSR-specific pieces are
implemented as custom nodes: LQ projection, block-0 conditioning, the one-step
streaming sampler, the Tiny Conditional Decoder, postprocessing, and an
optional SpargeAttn route for LCSA. ComfyUI stock nodes are used wherever possible.

> [!IMPORTANT]
> This is an independent community implementation, not the official FlashVSR
> repository. FlashVSR is designed primarily for 4x video super-resolution.
> Results, speed, and peak VRAM depend heavily on resolution, clip length,
> GPU, attention backend, and checkpoint dtype.

## Highlights

- Uses ComfyUI's stock Wan `MODEL` and native model patcher.
- Works with `BasicGuider`, `SamplerCustomAdvanced`, and stock Wan VAE decode.
- Loads FlashVSR safetensors from `ComfyUI/models/flashvsr`.
- Preserves FlashVSR's official causal temporal layout and LQ conditioning.
- Provides bounded overlap-context `streaming` sampling and a dense reference
  mode.
- Routes dense and cross-attention through the model's selected ComfyUI
  attention backend.
- Optionally executes FlashVSR's LCSA block mask with a separately installed
  SpargeAttn wheel.
- Includes a ComfyUI-managed Tiny Conditional Decoder with low-VRAM temporal
  batching.
- Includes crop-only, AdaIN, quarter-resolution wavelet, and full-resolution
  wavelet postprocessing.
- Includes an optional CUDA-event profiler for sampler diagnosis.

## Requirements

- A current ComfyUI installation with stock Wan support.
- Python 3.10 or newer; pytorch with CUDA 13.0 recommended.
- An NVIDIA CUDA GPU is strongly recommended.
- Enough system RAM and VRAM for the chosen resolution and model dtype; I tested this on an RTX 4050 6GB VRAM 16GB RAM laptop.
- FlashVSR v1.1 safetensors (not bundled).

## Installation

### ComfyUI Manager / Registry

Once this node is published to the Comfy Registry, search for
`ComfyUI FlashVSR Stock Wan`, install it, and restart ComfyUI.

### Git clone

From the `ComfyUI/custom_nodes` directory:

```bash
git clone https://github.com/Pizzawookiee/ComfyUI-FlashVSR-Stock.git
cd ComfyUI-FlashVSR-Stock
python -m pip install -r requirements.txt
```

For the Windows portable build, run the install with its embedded Python from
the `ComfyUI_windows_portable` directory:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-FlashVSR-Stock\requirements.txt
```

Restart ComfyUI after installation.

## Models

You can use the files I hosted
[pizzawookiee/FlashVSR-1.1](https://huggingface.co/pizzawookiee/FlashVSR-1.1/tree/main) and place
the FlashVSR-specific files in:

```text
ComfyUI/
└── models/
    └── flashvsr/
        ├── FlashVSR1_1-int8_convrot.safetensors.safetensors
        ├── LQ_proj_in.safetensors
        ├── Prompt.safetensors
        └── TCDecoder-fp16.safetensors
```

The loaders match component names rather than requiring those exact filenames,
so you can probably reuse your old safetensors FlashVSR files if you make sure that `FlashVSR1_1` is in the filename for example. 

The Wan 2.1 VAE is not FlashVSR-specific in this implementation. Put it in the
usual ComfyUI VAE directory, load it with ComfyUI's stock VAE loader and you can use stock ComfyUI VAE decode rather than TCDecoder:

```text
ComfyUI/models/vae/Wan2.1_VAE.safetensors
```

The older v1 checkpoint is not the intended main model for this integration.
Use `FlashVSR1_1.safetensors` or a compatible conversion of the v1.1 model.

## Optional SpargeAttn

`FlashVSR Sparge Attention` is optional. Install a
[SpargeAttn release](https://github.com/woct0rdho/SpargeAttn/releases) that
matches ComfyUI's Python, PyTorch, CUDA, and GPU architecture. The node imports
the wheel as `spas_sage_attn` only when the patch is used.

The standard requirements install the platform-appropriate Triton runtime used
by that optional route, but they do not install a SpargeAttn wheel. Dense
sampling does not import SpargeAttn.

Sparge is a private route for FlashVSR streaming self-attention: it executes
the LCSA mask built from `sparse_ratio` and `local_range`. Cross-attention,
dense fallback calls, and `full_video_dense` continue to use the model's
selected `ModelAttentionBackend`. Patch order therefore does not choose Sparge
instead of the general model backend.

If the wheel is missing or incompatible, remove the `FlashVSR Sparge
Attention` node and use a mask-capable ComfyUI attention backend. Sparge is not
bundled in this repository.

## Example workflow

Open [`workflows/FlashVSR_Stock_Wan_Workflow.json`](workflows/FlashVSR_Stock_Wan_Workflow.json)
in ComfyUI. Select your own input video and installed model files after loading
it.

The intended graph is:

1. Load a video and resize its frames to the desired output dimensions with
   any ComfyUI image-resize node.
2. Pass the already resized frames to `Prepare Video for FlashVSR`.
3. Load the FlashVSR v1.1 DiT, prompt tensor, and LQ projector.
4. Optionally apply `ModelAttentionBackend`, then optionally apply `FlashVSR
   Sparge Attention`.
5. Connect the model and FlashVSR components to `Configure FlashVSR Upscaling`.
6. Use `BasicGuider`, `FlashVSR One-Step Sampler`, and
   `SamplerCustomAdvanced`. The provided sampler owns the one-step streaming
   logic; a regular stock sampler cannot replace it.
7. Decode with either `FlashVSR Tiny Decode` or ComfyUI's stock Wan VAE.
8. Run `FlashVSR Postprocess` for color correction and to remove right/bottom alignment padding and
   optional temporal padding, then create/save the output video.

The initial latent is created by `Prepare Video for FlashVSR`. It has the
spatial and temporal shape required by the prepared video; the custom sampler
then fills it through FlashVSR's one-step model calls.

## Sampler settings

| Setting | Meaning | Practical guidance |
| --- | --- | --- |
| `sampling_mode=streaming` | Six latent frames are evaluated first; later calls recompute two overlap frames and append two or four new frames. LCSA is active. | Recommended default. Peak attention memory is bounded by the segment size rather than total clip length. |
| `sampling_mode=full_video_dense` | Projects and samples the whole video in one dense-attention call. | Reference/control mode. Usually faster for small clips but VRAM grows strongly with clip length and resolution. |
| `sparse_ratio` | Global LCSA selection budget. Larger values retain more eligible key blocks. | Higher may preserve more context but increases sparse work. It is not a percentage. Keep the default unless testing quality/performance. |
| `local_range` | Spatial neighborhood, measured in FlashVSR token blocks, from which LCSA may select. | Larger expands accessible spatial context and routing cost. Keep the default for the reference topology. |
| `query_block_chunk` | Maximum number of 128-token query blocks expanded per dense masked-attention call. `0` auto-selects a conservative value from free VRAM. | Lower reduces transient mask/attention memory but increases launches. Ignored by Sparge. |
| `new_latent_frames` | New frames appended per continuation call: `2` or `4`. | `4` normally reduces calls and boundary overhead; use `2` when memory is tight. |
| `profile_cuda_events` | Prints aggregate CUDA timings for LQ projection, model execution, routing, Sparge, and assembly. | Leave off for normal runs; profiling adds bookkeeping and a final synchronization. |

`streaming` bounds the model's temporal working set, but the prepared input,
final latent, and decoded/output frames still grow with video duration. Long
clips can therefore exhaust system RAM or VRAM outside the attention kernel.

## Decoder and postprocess settings

- For lowest Tiny Decoder VRAM, start with `temporal_batch_size=1`,
  `output_chunk_size=1`, `channels_last=false`, and `fuse_tgrow=false`.
- Increase `temporal_batch_size` only after confirming headroom. It can improve
  convolution utilization but increases activation memory.
- Stock Wan VAE decode is the quality reference. Tiny Decode is faster but is
  a different decoder and may not match the stock VAE exactly.
- `adain` is the inexpensive color correction.
- `wavelet_quarter_res` applies the five-level low-frequency correction at a
  reduced working resolution and is the practical wavelet mode.
- `wavelet_full_res` is the slowest and most memory-intensive correction.
- Postprocess always crops the right/bottom padding added for 128-pixel spatial
  alignment and removes padded tail frames.

## How this compares

These projects make different integration choices; none is universally best.
No speed ranking is claimed because results depend on hardware, resolution,
attention implementation, dtype, tiling, and decoder.

| Implementation | Integration style | Attention path | Model/VRAM behavior | Best fit |
| --- | --- | --- | --- | --- |
| **This repository** | Composable nodes around stock ComfyUI Wan, stock guider/sampler shell, stock VAE option, and `ModelAttentionBackend` | FlashVSR LCSA routing in `streaming`; any mask-capable dense backend, or optional Sparge for the selected blocks | Uses ComfyUI model patchers and `comfy.ops` for load/offload; explicit streaming and Tiny Decoder controls | Users who want FlashVSR to coexist with stock Wan workflows and backend patches |
| [Official FlashVSR](https://github.com/OpenImagingLab/FlashVSR) | Standalone reference pipeline | Official LCSA with the compiled Block-Sparse-Attention extension | Reference environment and checkpoint layout; not a native ComfyUI graph | Accuracy/reference validation and supported research setup |
| [1038lab ComfyUI-FlashVSR](https://github.com/1038lab/ComfyUI-FlashVSR) | Turnkey/all-in-one ComfyUI nodes | Optional SageAttention with fallback | Automatic model download, presets, tiling, and audio passthrough | Easiest self-contained setup |
| [ComfyUI-FlashVSR Ultra Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) | Custom FlashVSR pipeline | Sparse Sage; modified attention behavior | Long-video mode, tiled DiT/VAE, unload controls, full/tiny modes | Users prioritizing its integrated low-VRAM/tiled workflow |
| [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) | FlashVSR integrated into the wrapper's custom Wan model/sampling ecosystem | Wrapper attention patches and FlashVSR conditioning | Broad WanVideoWrapper model-management and optimization stack | Existing WanVideoWrapper workflows |

The official project specifically warns that replacing LCSA with plain dense
attention can reduce quality at high resolution. This implementation's
`full_video_dense` mode is therefore a diagnostic/control path, not the
recommended quality-equivalent replacement for `streaming` LCSA.

## Troubleshooting

### Models do not appear

Confirm that the files are `.safetensors`, restart ComfyUI, and place all
FlashVSR-specific weights under lowercase `ComfyUI/models/flashvsr`. Load the
Wan VAE through the stock VAE loader; that VAE should reside in the usual `ComfyUI/models/vae`.

### Out of memory

Use `streaming`, set `new_latent_frames=2`, use a memory-efficient attention
backend, keep Tiny Decode at `temporal_batch_size=1`, and test a shorter or
lower-resolution clip. An INT8/ConvRot DiT can reduce model weight residency,
but activations, LQ projection, attention, latent storage, and decode output
still require memory.

### Attention backend rejects the mask

Streaming dense fallback requires a backend that accepts an arbitrary
per-head attention mask. Switch `ModelAttentionBackend` to a compatible mode,
or use the optional FlashVSR Sparge node included in this repo.

### First run is slower

CUDA kernels and optional Triton/Sparge components may initialize or compile
on their first compatible call. Compare repeated runs only after the workflow
has completed successfully at least once.

## Release notes — 0.20.0

- Public-release packaging and Comfy Registry metadata.
- Added the example stock-Wan workflow after a PII/secret audit.
- Added the registry publishing GitHub Action.
- Reworked installation, workflow, tuning, attention, and implementation
  comparison documentation.
- Sampling and decoding behavior are unchanged from the validated 0.19.9
  baseline: `topk(sorted=False)` LCSA selection and the expanded CUDA profiler
  remain available.

## Credits and license

This project borrows from components described by
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) and
[ComfyUI](https://github.com/Comfy-Org/ComfyUI) APIs. Safetensors naming and
packaging follow the files published at
[pizzawookiee/FlashVSR-1.1](https://huggingface.co/pizzawookiee/FlashVSR-1.1/tree/main).
The optional sparse executor calls the separately distributed
[SpargeAttn](https://github.com/woct0rdho/SpargeAttn) library. See
[`NOTICE`](NOTICE) for attribution details.


