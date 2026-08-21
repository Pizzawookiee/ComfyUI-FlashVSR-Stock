If you're having OOM errors with other ComfyUI FlashVSR nodes, try this one.

A 4x spatial upscale of a 10 second video takes about 6 min on a RTX 4050 6GB VRAM 16GB RAM machine with 'streaming_faithful_lowvram' setting in bundled sampler node.

Use 'streaming_faithful_lowvram' in bundled sampler node if bounded by VRAM (i.e. OOM or major slowdown), as KV cache can be costly to offload to RAM. Or, use 'streaming' mode which drops the cache entirely and can be good enough.

# ComfyUI FlashVSR — Stock Wan

FlashVSR v1.1 video super-resolution built around ComfyUI's stock Wan model,
model patching, sampling, VAE, safetensors loading, dynamic VRAM management,
and attention-backend system.

This project is intended for users who want FlashVSR as a composable ComfyUI
workflow rather than an all-in-one pipeline. The FlashVSR-specific pieces are
implemented as custom nodes: LQ projection, block-0 conditioning, the one-step
streaming sampler, the Tiny Conditional Decoder, postprocessing, and an
optional SpargeAttn route for LCSA.

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
- Provides bounded overlap-context sampling, two paper-layout streaming modes
  with sliding DiT KV caches, and a dense reference mode.
- Uses stock ComfyUI Wan Q, K, and V projections by default, including stock
  Comfy-Kitchen execution for INT8 ConvRot checkpoints. An output-validated
  shared-activation projection remains available as an experimental option.
- Stores faithful temporal history as independently selectable INT8, hybrid,
  or floating carriers. Bounded per-block GPU residency uses available VRAM
  while the remaining cache stays on CPU.
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
- The Comfy-Kitchen build bundled with that ComfyUI installation. It is used
  normally by ComfyUI for compatible quantized checkpoints; the optional
  experimental shared-QKV path is capability-checked.
- Python 3.10 or newer.
- An NVIDIA CUDA GPU is strongly recommended.
- Enough system RAM and VRAM for the chosen resolution and model dtype.
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

Download the safetensors from
[pizzawookiee/FlashVSR-1.1](https://huggingface.co/pizzawookiee/FlashVSR-1.1/tree/main) and place
the FlashVSR-specific files in:

```text
ComfyUI/
└── models/
    └── flashvsr/
        ├── FlashVSR1_1.safetensors
        ├── LQ_proj_in.safetensors
        ├── Prompt.safetensors
        └── TCDecoder.safetensors
```

The loaders match component names rather than requiring those exact filenames,
so compatible dtype-converted files such as `FlashVSR1_1-int8_convrot.safetensors`
or `TCDecoder-fp16.safetensors` can also appear in the menus.

The Wan 2.1 VAE is not FlashVSR-specific in this implementation. Put it in the
usual ComfyUI VAE directory and load it with ComfyUI's stock VAE loader:

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
in ComfyUI. The included workflow contains no personal paths, credentials, or
other PII. Select your own input video and installed model files after loading
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
8. Run `FlashVSR Postprocess` to remove right/bottom alignment padding and
   optional temporal padding, then create/save the output video.

The initial latent is created by `Prepare Video for FlashVSR`. It has the
spatial and temporal shape required by the prepared video; the custom sampler
then fills it through FlashVSR's one-step model calls.

## Sampler settings

| Setting | Meaning | Practical guidance |
| --- | --- | --- |
| `sampling_mode=full_video_dense` | Projects and samples the whole video in one dense-attention call. | Reference/control mode. Usually faster for small clips but VRAM grows strongly with clip length and resolution. |
| `sampling_mode=streaming` | Six latent frames are evaluated first; later calls recompute two overlap frames and append two or four new frames. There is no DiT KV cache. | Proven compatibility mode. Peak attention memory is bounded by segment size. |
| `sampling_mode=streaming_faithful_full` | Six-frame prefill followed by exactly two new frames per call. Every Wan block retains a sliding six-frame post-RoPE K/V history. | Closest mode to the paper. Cache storage automatically falls back to CPU when the complete cache does not fit conservatively in VRAM. Highest RAM and transfer cost. |
| `sampling_mode=streaming_faithful_lowvram` | Uses the faithful two-frame continuation layout and retains the nearest two historical frames in every Wan block. | Low-VRAM compromise that preserves immediate temporal context throughout the DiT. Roughly one third of the full six-frame cache, but with less long-range history than paper inference. |
| `sparse_ratio` | Global LCSA selection budget. Larger values retain more eligible key blocks. | Higher may preserve more context but increases sparse work. It is not a percentage. Keep the default unless testing quality/performance. |
| `local_range` | Spatial neighborhood, measured in FlashVSR token blocks, from which LCSA may select. | Larger expands accessible spatial context and routing cost. Keep the default for the reference topology. |
| `query_block_chunk` | Maximum number of 128-token query blocks expanded per dense masked-attention call. `0` auto-selects a conservative value from free VRAM. | Lower reduces transient mask/attention memory but increases launches. Ignored by Sparge. |
| `new_latent_frames` | New frames appended per continuation call: `2` or `4`. | `2` is the default and is mandatory for faithful modes. `4` is available only as a legacy-streaming throughput option. |
| `qkv_projection` | `stock` uses ComfyUI's Wan projections; `shared_int8_experimental` reuses one ConvRot activation quantization. | Keep `stock` for release-quality output. The experimental path is validated against stock before use and does not determine cache size. |
| `cache_format` | Faithful-cache storage: `int8`, `hybrid`, or `float`. | `int8` is smallest; `hybrid` keeps K floating and quantizes V; `float` is the quality/control format. Independent of model dtype and QKV projection. |
| `cache_vram_policy` | Per-block cache placement: `cpu`, `conservative`, `balanced`, `aggressive`, or `custom`. | Start with `conservative`. More GPU-resident blocks reduce PCIe transfer but leave less attention workspace. |
| `cache_vram_budget_mb` | Explicit cache VRAM cap for the `custom` policy. | The allocator still keeps a safety reserve and falls individual blocks back to CPU. |
| `profile_cuda_events` | Prints aggregate CUDA timings for LQ projection, KV staging and writes, model execution, routing, Sparge, and assembly. | Leave off for normal runs; profiling adds bookkeeping and a final synchronization. |

`streaming` bounds the model's temporal working set, but the prepared input,
final latent, and decoded/output frames still grow with video duration. Long
clips can therefore exhaust system RAM or VRAM outside the attention kernel.

The full faithful cache is a three-slot chronological ring containing two
latent frames per slot. The low-VRAM cache uses one two-frame slot in every
Wan block. Both reach a fixed maximum after prefill, so peak cache memory does
not grow with video duration. When stored on CPU, however, the cache is staged
once per participating block and continuation; total PCIe traffic and runtime
therefore grow with clip length. On a 6 GB GPU, start with
`streaming_faithful_lowvram`. The full mode can use several GiB of system RAM
at HD output resolutions.

Cache precision is independent of the Wan checkpoint and QKV projection path.
The default `int8` format stores K and V with per-token/per-head scales. The
`hybrid` format keeps post-RoPE K in model precision while compacting V, and
`float` keeps both tensors in model precision. Every format expands or copies
into the reusable one-block GPU staging buffer before the selected attention
backend runs, preserving ModelAttentionBackend and current Sparge
compatibility.

ComfyUI Dynamic VRAM remains the sole owner of model-weight offloading. The
FlashVSR cache manager handles only mutable temporal carriers, which are not
model parameters and cannot be registered safely with AIMDO. Cache policies
therefore use an explicit safety reserve, keep only complete per-block caches
on GPU, and retain the remainder on CPU. If one GPU cache allocation fails,
only that block falls back to CPU.

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
| **This repository** | Composable nodes around stock ComfyUI Wan, stock guider/sampler shell, stock VAE option, and `ModelAttentionBackend` | Logical 128×128 LCSA routing; overlap streaming or two-frame faithful streaming with all-block KV caching; optional Sparge | Uses ComfyUI model patchers and `comfy.ops`; stock QKV is the default, cache precision is independent, and bounded hybrid CPU/GPU residency reduces offload traffic | Users who want FlashVSR to coexist with stock Wan workflows and backend patches |
| [Official FlashVSR](https://github.com/OpenImagingLab/FlashVSR) | Standalone reference pipeline | Official LCSA with the compiled Block-Sparse-Attention extension | Reference environment and checkpoint layout; not a native ComfyUI graph | Accuracy/reference validation and supported research setup |
| [1038lab ComfyUI-FlashVSR](https://github.com/1038lab/ComfyUI-FlashVSR) | Turnkey/all-in-one ComfyUI nodes | Optional SageAttention with fallback | Automatic model download, presets, tiling, and audio passthrough | Easiest self-contained setup |
| [ComfyUI-FlashVSR Ultra Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) | Custom FlashVSR pipeline | Sparse Sage; modified attention behavior | Long-video mode, tiled DiT/VAE, unload controls, full/tiny modes | Users prioritizing its integrated low-VRAM/tiled workflow |
| [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) | FlashVSR integrated into the wrapper's custom Wan model/sampling ecosystem | Wrapper attention patches and FlashVSR conditioning | Broad WanVideoWrapper model-management and optimization stack | Existing WanVideoWrapper workflows |

The official project specifically warns that replacing LCSA with plain dense
attention can reduce quality at high resolution. This implementation's
`full_video_dense` mode is therefore a diagnostic/control path, not the
recommended quality-equivalent replacement for LCSA streaming. Use
`streaming_faithful_full` when comparing most closely with the paper method.

## Troubleshooting

### Models do not appear

Confirm that the files are `.safetensors`, restart ComfyUI, and place all
FlashVSR-specific weights under lowercase `ComfyUI/models/flashvsr`. Load the
Wan VAE through the stock VAE loader from `ComfyUI/models/vae`.

### Out of memory

Use `streaming` or `streaming_faithful_lowvram`, keep
`new_latent_frames=2`, use a memory-efficient attention backend, keep Tiny
Decode at `temporal_batch_size=1`, and test a shorter or lower-resolution
clip. Faithful CPU cache offload reduces VRAM but still consumes system RAM.
The default compact cache works independently of checkpoint dtype. If fine
detail is sensitive to cache quantization, try `hybrid`, then `float`. Reduce
the cache VRAM policy before lowering resolution if an aggressive policy
causes an OOM.

### Attention backend rejects the mask

Streaming dense fallback requires a backend that accepts an arbitrary
per-head attention mask. Switch `ModelAttentionBackend` to a compatible mode,
or install and use the optional FlashVSR Sparge patch.

### First run is slower

CUDA kernels and optional Triton/Sparge components may initialize or compile
on their first compatible call. Compare repeated runs only after the workflow
has completed successfully at least once.

## Release notes — 0.31.0

- Restored stock ComfyUI Wan Q/K/V projection as the default and corrected
  the experimental CUTLASS FP32-bias ABI.
- Decoupled QKV execution from faithful-cache precision; compact cache storage
  works with stock projection and any supported checkpoint dtype.
- Added INT8, float-K/INT8-V hybrid, and floating cache formats. INT8 carrier
  scales are now per token and attention head.
- Added bounded per-block CPU/GPU cache placement with explicit VRAM policies
  and individual-block CPU fallback.
- Added experimental projection output validation and clearer cache residency
  and transfer diagnostics.

## Release notes — 0.3.0

- Added a clean-room shared ConvRot INT8 QKV execution path around stock Wan
  self-attention. Q/K/V remain separate ComfyUI-managed parameters.
- Added sequential Dynamic VRAM weight leases as the default accelerated path
  and a resident shared-carrier path when all three weights are already
  resident. Both currently use three projection launches.
- Added compact row-wise INT8 post-RoPE K/V carriers for both faithful modes,
  with bounded per-block expansion for existing attention backends.
- Made faithful cache placement CPU-first whenever ComfyUI Dynamic VRAM is
  active; no second model-weight offloader is introduced.
- Made cache-ring updates transactional so an interrupted block traversal
  cannot commit a partial temporal step.
- Added explicit capability/fallback diagnostics and QKV CUDA profiler stages.

## Release notes — 0.20.2

- Changed `streaming_faithful_lowvram` from a six-frame cache in only the
  final ten Wan blocks to a two-frame cache in every Wan block.
- Low-VRAM continuations now preserve immediate temporal context throughout
  the DiT, eliminating the systematic boundary blur caused by uncached early
  blocks while retaining approximately the previous cache footprint on a
  30-block Wan model.
- The low-VRAM prefill stores the final two frames of the six-frame prefill,
  ensuring that the first continuation receives the nearest available
  history.

## Release notes — 0.20.1

- Corrected LCSA routing to score logical 128-query by 128-key
  `2×8×8` windows. Sparge conversion to physical kernel geometry now happens
  only after logical routing.
- Changed the continuation default from four new latent frames to two.
- Added `streaming_faithful_full`, with a six-frame sliding post-RoPE K/V
  cache in every Wan block.
- Added `streaming_faithful_lowvram`, which caches the final ten Wan blocks.
- Added conservative automatic GPU/CPU cache placement, reusable one-block
  staging buffers, cache lifecycle validation, and KV transfer profiling.
- Faithful cache modes require a single conditional pass through BasicGuider
  or ComfyUI's active CFG=1 optimization.

## Previous release — 0.20.0

- Public-release packaging and Comfy Registry metadata.
- Added the example stock-Wan workflow after a PII/secret audit.
- Added the registry publishing GitHub Action.
- Reworked installation, workflow, tuning, attention, and implementation
  comparison documentation.
- Sampling and decoding behavior are unchanged from the validated 0.19.9
  baseline: `topk(sorted=False)` LCSA selection and the expanded CUDA profiler
  remain available.

## Publishing checklist

Before the first Comfy Registry publish:

1. Confirm the `Pizzawookiee` publisher exists in the Comfy Registry and that
   the publishing token has permission to publish `comfyui-flashvsr-stock`.
2. Optionally add a repository-relative icon and update `Icon`; do not use the
   placeholder example URL from the Registry template.
3. Add the repository secret `REGISTRY_ACCESS_TOKEN` in GitHub Actions.
4. Commit a `pyproject.toml` change on `main`, or manually dispatch
   `.github/workflows/publish_action.yml`.

## Credits and license

This project reimplements components described by
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) and uses stock
[ComfyUI](https://github.com/Comfy-Org/ComfyUI) APIs. Safetensors naming and
packaging follow the files published at
[pizzawookiee/FlashVSR-1.1](https://huggingface.co/pizzawookiee/FlashVSR-1.1/tree/main).
The optional sparse executor calls the separately distributed
[SpargeAttn](https://github.com/woct0rdho/SpargeAttn) library. See
[`NOTICE`](NOTICE) for attribution details.

The repository code is licensed under `GPL-3.0-only`. This conservative choice
covers the GPLv3 implementation references used during development while still
allowing Apache-2.0 source material to be incorporated under GPLv3's terms.
Model weights and separately installed dependencies retain their own licenses
and terms.
