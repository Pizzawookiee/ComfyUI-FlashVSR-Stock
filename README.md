If you're having OOM errors with other ComfyUI FlashVSR nodes, try this one.

A spatial upscale of 2x each side of a 10 second 0.4 MP video takes 5 min on a RTX 4050 6GB VRAM 16GB RAM machine with 'streaming_faithful_lowvram' sampler mode in bundled sampler node.

Switch to 'streaming_faithful_lowvram' in bundled sampler node for more faithful implementation while bounded by VRAM (i.e. OOM or major slowdown); default is 'streaming' as that drops KV cache entirely and uses the least RAM with good enough results.

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
  or floating carriers. CPU is always authoritative; an optional AIMDO VBAR
  can retain evictable GPU mirrors without a fixed cache-VRAM budget.
- Routes dense and cross-attention through the model's selected ComfyUI
  attention backend.
- Optionally executes FlashVSR's LCSA block mask with a separately installed
  SpargeAttn wheel.
- Includes a ComfyUI-managed Tiny Conditional Decoder with low-VRAM temporal
  batching.
- Includes crop-only, AdaIN, quarter-resolution wavelet, and full-resolution
  wavelet postprocessing.
- Includes an optional CUDA-event profiler for sampler diagnosis.
- Faithful INT8 caches paired with the private Sparge route reuse cached K
  summaries and dequantize carriers directly into final HND kernel buffers.
- Compatible Sparge lower kernels can consume native block-INT8 K and, on
  FP8 architectures, directly materialized transposed-FP8 V without complete
  floating-point history buffers.
- Bounded asynchronous cache write-through preserves AIMDO-controlled
  residency; no Wan block is manually selected to remain on the GPU.
- Includes an independent optional CUDA-event profiler for TCDecoder.

## Requirements

- A current ComfyUI installation with stock Wan support.
- The Comfy-Kitchen build bundled with that ComfyUI installation. It is used
  normally by ComfyUI for compatible quantized checkpoints; the optional
  experimental shared-QKV path is capability-checked.
- Python 3.10 or newer.
- An NVIDIA CUDA GPU is strongly recommended.
- Enough system RAM and VRAM for the chosen resolution and model dtype.
- FlashVSR v1.1 safetensors (not bundled).

The experimental AIMDO cache-residency backend additionally requires a
current ComfyUI build with AIMDO available and enabled. It is optional: the
default `cpu` backend does not import or initialize `comfy_aimdo`.

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

The optional Wan 2.1 VAE is not FlashVSR-specific in this implementation. Put it in the
usual ComfyUI VAE directory and load it with ComfyUI's stock VAE loader:

```text
ComfyUI/models/vae/Wan2.1_VAE.safetensors
```

Note that upscaling videos with this repo does not require the stock Wan 2.1 VAE.

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

## Optional modified H3-Optimizations library

The repo (https://github.com/Zironic/H3-Optimizations) packages a modified comfy-kitchen library which is
compatible with this repo. Benefits include further reduction of BF16/FP16 intermediats and better support for streamed execution.

To install, you must
a) copy the dll from that repo's native/bin into Comfy's models/flashvsr
b) use the `FlashVSR Kitchen Sparse Attention` node in place of the Sparge node.


> [!WARNING]
> The v0.34 native compact-cache adapter targets RTX 4000-series GPUs (SM89).
> Sparge exposes related lower ABIs for SM80,
> SM86, SM87, SM90, SM100, SM120, and SM121, and v0.34 parameterizes their
> documented block/V formats, but these non-RTX-4000 paths are **untested by
> this project**.

Compatible SM89-family kernels receive native block-INT8 K and directly
materialized transposed-FP8 V. SM80/86/87 use native K but retain FP16 V
because that is what Sparge's Ampere kernel accepts. SM90 uses its separate
64-query/128-key FP8 ABI.

## Example workflow

Open [`workflows/FlashVSR_Stock_Wan_Workflow.json`](workflows/FlashVSR_Stock_Wan_Workflow.json)
in ComfyUI. Select your own input video and installed model files after loading
it.

The intended graph is:

1. Load a video.
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

The 0.36 update changes `Prepare Video for FlashVSR` to use lazy upscaling which lowers RAM usage.
Previously the node did not handle upscaling, requiring you to upscale frames with a Comfy upscaling node.
Starting with 0.36 update, beware of stacking `Prepare Video for FlashVSR` upscaling with other ComfyUI image upscale nodes.

## Sampler settings

| Setting | Meaning | Practical guidance |
| --- | --- | --- |
| `sampling_mode=full_video_dense` | Projects and samples the whole video in one dense-attention call. | Reference/control mode. Usually faster for small clips but VRAM grows strongly with clip length and resolution. |
| `sampling_mode=streaming` | Six latent frames are evaluated first; later calls recompute two overlap frames and append two or four new frames. There is no DiT KV cache. | Proven compatibility mode, generally the fastest sampling mode. Peak attention memory is bounded by segment size. |
| `sampling_mode=streaming_faithful_full` | Six-frame prefill followed by exactly two new frames per call. Every Wan block retains a sliding six-frame post-RoPE K/V history. | Closest mode to the paper. CPU residency is the reliable default; optional AIMDO residency can avoid transfers for pages that remain mapped. Highest RAM and transfer cost. |
| `sampling_mode=streaming_faithful_lowvram` | Uses the faithful two-frame continuation layout and retains the nearest two historical frames in every Wan block. | Low-VRAM compromise that preserves immediate temporal context throughout the DiT. Roughly one third of the full six-frame cache, but with less long-range history than paper inference. |
| `sparse_ratio` | Global LCSA selection budget. Larger values retain more eligible key blocks. | Higher may preserve more context but increases sparse work. It is not a percentage. Keep the default unless testing quality/performance. |
| `local_range` | Spatial neighborhood, measured in FlashVSR token blocks, from which LCSA may select. | Larger expands accessible spatial context and routing cost. Keep the default for the reference topology. |
| `query_block_chunk` | Maximum number of 128-token query blocks expanded per dense masked-attention call. `0` auto-selects a conservative value from free VRAM. | Lower reduces transient mask/attention memory but increases launches. Ignored by Sparge. |
| `new_latent_frames` | New frames appended per continuation call: `2` or `4`. | `2` is the default and is mandatory for faithful modes. `4` is available only as a legacy-streaming throughput option. |
| `qkv_projection` | `stock` uses ComfyUI's Wan projections; `shared_int8_experimental` reuses one ConvRot activation quantization. | Keep `stock` for release-quality output. The experimental path is validated against stock before use and does not determine cache size. |
| `cache_format` | Faithful-cache storage: `int8`, `hybrid`, or `float`. | `int8` is smallest; `hybrid` keeps K floating and quantizes V; `float` is the quality/control format. Independent of model dtype and QKV projection. |
| `cache_residency_backend` | `cpu` always stages from the authoritative system-RAM cache. `aimdo_experimental` adds a dedicated, deprioritized AIMDO VBAR as an evictable GPU mirror (i.e. automatic GPU staging if there is VRAM available) | Start with `cpu` for reliability. AIMDO resident hits avoid PCIe staging; misses, stale mappings, initialization failures, and per-access faults fall back to the same CPU data. |
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
`float` keeps both tensors in model precision. With `int8` plus `FlashVSR
Sparge Attention`, cached K summaries drive routing.

ComfyUI Dynamic VRAM remains the sole owner of model-weight offloading. The
FlashVSR cache manager handles only mutable temporal carriers. In `cpu` mode,
every carrier remains in system RAM. In `aimdo_experimental` mode, FlashVSR
creates a separate lower-priority packed VBAR; CPU remains authoritative and
each VBAR mapping is validated before use. Continuation writes are
transactional and write through to a resident mirror when possible. A page
fault miss falls back for that access, while an AIMDO initialization or API
failure disables the mirror for the rest of the run. This avoids a second
hand-written model-weight offloader and lets AIMDO reclaim cache pages before
higher-priority model pages.

INT8 continuation writes use a two-slot pinned staging ring and dedicated CUDA
transfer stream. Completed data is copied into ordinary CPU RAM by bounded
background workers before the cache transaction commits. AIMDO still chooses
residency, and CPU remains a complete fallback. The full cache is never pinned.
Starting ComfyUI with `--disable-pinned-memory`, or any pinned allocation
failure, automatically restores synchronous write-through.

## Decoder and postprocess settings

- For lowest Tiny Decoder VRAM, start with `temporal_batch_size=1`,
  `output_chunk_size=1`, `channels_last=false`, and `fuse_tgrow=false`.
- Increase `temporal_batch_size` only after confirming headroom. It can improve
  convolution utilization but increases activation memory.
- Stock Wan VAE decode is the quality reference. Tiny Decode is faster but is
  a different decoder and may not match the stock VAE exactly.
- `profile_cuda_events` on `FlashVSR Tiny Decode` prints resolution-grouped
  TCDecoder stages, wall time, throughput, and peak allocated/reserved VRAM.
  Leave it disabled normally because it performs a final synchronization.
- `compile_memblocks` is an experimental `torch.compile`/Inductor path for
  static TCDecoder MemBlocks only. It has a slower first invocation and
  automatically falls back to eager execution after a setup/runtime failure.
- `color_device=auto` uses bounded FP16 CUDA chunks when CUDA is available;
  choose `cpu` for the compatibility path. `inplace_correction=true` avoids a
  second complete output-video allocation but should be disabled if another
  workflow branch consumes the uncorrected decoder output.
- `profile_stages` on `FlashVSR Postprocess` prints transfer, downsample,
  lowpass, upsample/correction, clamp and output timing plus complete wall time.
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
detail is sensitive to cache quantization, try `hybrid`, then `float`. If the
experimental AIMDO mirror contributes to memory pressure, switch
`cache_residency_backend` to `cpu` before lowering resolution.

### Attention backend rejects the mask

Streaming dense fallback requires a backend that accepts an arbitrary
per-head attention mask. Switch `ModelAttentionBackend` to a compatible mode,
or install and use the optional FlashVSR Sparge patch.

### First run is slower

CUDA kernels and optional Triton/Sparge components may initialize or compile
on their first compatible call. Compare repeated runs only after the workflow
has completed successfully at least once.

## Release notes — see CHANGELOG.md

## Credits and license

This project reimplements components described by
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) and uses stock
[ComfyUI](https://github.com/Comfy-Org/ComfyUI) APIs. Safetensors naming and
packaging follow the files published at
[pizzawookiee/FlashVSR-1.1](https://huggingface.co/pizzawookiee/FlashVSR-1.1/tree/main).
The optional sparse executor calls the separately distributed
[SpargeAttn](https://github.com/woct0rdho/SpargeAttn) library.
This repo can also use the modified comfy-kitchen library in [H3-Optimizations] (https://github.com/Zironic/H3-Optimizations).
See [`NOTICE`](NOTICE) for attribution details.

The repository code is licensed under `GPL-3.0-only`. This conservative choice
covers the GPLv3 implementation references used during development while still
allowing Apache-2.0 source material to be incorporated under GPLv3's terms.
Model weights and separately installed dependencies retain their own licenses
and terms.
