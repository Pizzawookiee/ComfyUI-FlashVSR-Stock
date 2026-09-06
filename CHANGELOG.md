# Changelog

## 0.37.0
- Added streamed Kitchen attention output projection to reduce large pre-output intermediate allocations.
- Added additional cached routing metadata for sparse attention.
- Added cached spatial H/W RoPE construction, while preserving per-segment temporal RoPE updates.
- Added Triton fused LayerNorm + AdaLN modulation for Wan DiT rows, with automatic fallback to the existing PyTorch path.
- Added Triton fused gated-residual updates to reduce repeated elementwise kernel launches.
- Added steady sparse-mask threshold reuse across continuation chunks to reduce repeated top-k work.
- Improved TCDecoder performance with split recurrent convolutions, low-resolution TGrow execution,
channels-last support and fused cuDNN Conv/Add/ReLU paths.
- Improved LQ projector memory efficiency with tiled im2col + GEMM execution and bounded Conv2 working memory.
- Reduced unnecessary intermediate tensors and allocation churn throughout the streaming inference path.
- Continued optimizing faithful low-VRAM execution for 6 GB-class GPUs without changing the model architecture or checkpoint format.

## 0.36.0
- Tiling and chunking of various layers for reduced memory pressure
- Eliminate higher precision intermediates where possible, with focus on int8
- Support custom comfy kitchen used in https://github.com/Zironic/H3-Optimizations,
  copy the dll in that repo's native/bin and paste it in Comfy install's models/flashvsr,
  then use 'FlashVSR Kitchen Sparse Attention' in place of 'FlashVSR Sparge Attention' node
- 'Prepare Video for FlashVSR' node now adds lazy upscaling,
  i.e. only upscale low res frames when necessary rather than all at once with stock image upscale nodes.
  Should help reduce RAM usage in some situtations.
  That node now has new 'scale_multiplier' setting, indepedent of other comfy upscaling nodes.


## 0.35.0

- Corrected asynchronous cache destinations so background CPU write-through
  remains mutable when ComfyUI executes the sampler under inference mode.
- Builds LCSA K summaries directly from the live continuation projection,
  removing the immediate INT8 dequantize-and-reread pass used in v0.34.
- Reduced native FP8 V preparation allocations by using one contiguous partial
  maximum workspace and an explicit output reduction buffer.
- Added optional TCDecoder MemBlock `torch.compile`/Inductor execution with
  static shapes, first-use diagnostics, and automatic eager fallback. The
  default remains eager for Dynamic VRAM compatibility.
- Reworked FlashVSR Postprocess to support bounded GPU FP16 or CPU correction,
  optional in-place output, and detailed per-stage profiling.
- Quarter-resolution wavelet correction now downsamples content and style
  before subtraction, avoiding a full-resolution difference allocation.


## 0.34.0

- Added an experimental native compact-cache Sparge adapter. Faithful INT8 K
  is converted directly into Sparge's block-INT8 layout using one fixed
  post-RoPE prefill reference mean per Wan layer. On the FP8 Sparge families,
  V is materialized directly from row-INT8 history into the transposed FP8
  kernel layout without a full FP16 history tensor.
- Parameterized native dispatch for Sparge's SM80-family, SM89-family and SM90
  lower ABIs. RTX 4000 / SM89 is the development target; all non-RTX-4000
  paths are explicitly marked untested and retain automatic v0.33 fallback.
- Replaced synchronous INT8 cache write-through with a bounded two-slot pinned
  staging ring, dedicated CUDA transfer stream and ordinary-RAM authoritative
  cache. AIMDO still decides residency; no manual block placement or VRAM
  budget was introduced.
- Added cache-write enqueue, transfer, wait, byte-count and AIMDO fault
  profiler diagnostics.
- Added optional TCDecoder CUDA-event profiling grouped by spatial resolution,
  including conditioning transfer, pixel unshuffle, convolution, MemBlock,
  TGrow, state, crop/output, wall-time, throughput and peak-memory reporting.
- Kept all v0.33 compact materialization and public Sparge routes as guarded
  compatibility fallbacks.

## 0.33.0

- Added cached K routing summaries for faithful INT8 continuations.
- Added a compact cache descriptor path for FlashVSR Sparge Attention.
- Dequantized cached INT8 K/V directly into final block-ordered HND buffers,
  removing full floating-point cache staging and its second layout copy.
- Preserved the v0.32 staging route for float/hybrid caches, dense attention,
  initial prefill, and compatibility fallback.
- Extended the CUDA profiler with compact-cache transfer/dequantization and
  detailed Sparge quantization, LUT, V-layout, and lower-kernel timings.

## 0.32.0

- Replaced `cache_vram_policy` and `cache_vram_budget_mb` with
  `cache_residency_backend` (`cpu` or `aimdo_experimental`).
- Added a dedicated, deprioritized, packed AIMDO VBAR for evictable faithful
  cache mirrors while retaining an authoritative CPU copy of every carrier.
- Added signature and content-generation validation, transactional
  continuation writes, CPU fallback on individual fault misses, and a
  run-wide CPU fallback after AIMDO initialization/API failures.
- Added AIMDO access, hit, rehydrate, miss, fallback, write-through,
  residency, populated-byte, and host-fault-time diagnostics.
- Kept cache precision and Wan QKV projection independent from residency.

## 0.31.0

- Restored stock ComfyUI Wan Q/K/V projection as the default streaming path
  after identifying an FP16/BF16-bias ABI mismatch in the experimental direct
  CUTLASS projection.
- Corrected the experimental CUTLASS path to supply FP32 bias, require SM80 or
  newer, and validate a small Q/K/V output sample against stock projection
  before continuing.
- Decoupled faithful cache precision from QKV projection. Stock ComfyUI INT8
  ConvRot models can now use compact cache carriers without enabling shared
  QKV projection.
- Added `int8`, `hybrid`, and `float` cache formats. Hybrid retains K in model
  precision and stores V as INT8; compact carriers now scale each token/head
  independently for better attention fidelity.
- Added bounded per-block GPU cache residency with CPU, conservative,
  balanced, aggressive, and custom VRAM policies. Individual allocation
  failures fall back only the affected block to CPU.
- Expanded sampler tooltips and cache diagnostics with format, residency, and
  remaining transfer estimates.

## 0.3.0

- Added a clean-room shared ConvRot INT8 QKV projection path for stock Wan.
  Compatible Q/K/V projections reuse one activation rotation/quantization
  while remaining separate ComfyUI-managed parameters.
- Added sequential Dynamic VRAM leases as the low-memory default and a
  resident shared-carrier path when all three weights are already resident.
  Both currently use three projection launches.
- Added compact row-wise INT8 post-RoPE K/V cache carriers for faithful modes,
  with bounded per-block expansion for ModelAttentionBackend and Sparge.
- Made faithful carrier storage CPU-first under ComfyUI Dynamic VRAM and kept
  model weight offloading entirely under ComfyUI/AIMDO.
- Made faithful cache updates transactional across a complete model chunk.
- Added guarded stock-projection fallback for non-INT8 checkpoints,
  incompatible layouts, weight patches, missing Comfy-Kitchen CUDA symbols,
  and unsupported projection shapes.
- Added CUDA profiler stages for shared ConvRot quantization, Q/K/V GEMMs,
  and Q/K normalization plus RoPE.

## 0.20.2

- Reworked `streaming_faithful_lowvram` to cache the nearest two historical
  latent frames in every Wan block instead of six frames in only the final
  ten blocks.
- Preserved approximately the same cache footprint and aggregate attention
  workload for a 30-block Wan model while restoring temporal context to early
  transformer layers.
- Corrected low-VRAM prefill selection to retain the final two frames of the
  initial six-frame segment.
- Updated cache diagnostics and sampler documentation for the new behavior.

## 0.20.1

- Corrected LCSA routing geometry to logical 128-query by 128-key
  `2x8x8` blocks, with backend-specific Sparge conversion applied afterward.
- Changed the default continuation size from four new latent frames to two.
- Added `streaming_faithful_full`: six-frame prefill, two-frame
  continuations, and a six-frame sliding post-RoPE K/V cache in every Wan
  block.
- Added `streaming_faithful_lowvram`, which applies the same temporal layout
  and caches the final ten Wan blocks.
- Added conservative automatic GPU/CPU cache placement, reusable block-local
  GPU staging, cache-ring lifecycle checks, strict single-pass guider
  validation, and K/V transfer profiling.
- Preserved `full_video_dense` and overlap-context `streaming` for comparison
  and backward compatibility.

## 0.20.0

- Prepared the project for public GitHub and Comfy Registry distribution.
- Selected `GPL-3.0-only` as the conservative project license after reviewing
  the Apache-2.0 and GPLv3 implementation references used during development.
- Set the Comfy Registry publisher to `Pizzawookiee`, the display name to
  `ComfyUI-FlashVSR-Stock`, and the model source to the Pizzawookiee
  FlashVSR-1.1 Hugging Face repository.
- Added an audited stock-Wan example workflow.
- Added registry publishing automation.
- Replaced development-history-first documentation with installation,
  workflow, tuning, attention, troubleshooting, and implementation-comparison
  guidance.
- Retained the validated 0.19.9 runtime behavior, including
  `topk(sorted=False)` LCSA threshold selection and the expanded CUDA profiler.

## 0.19.9

- Restored the proven `topk(sorted=False)` plus `amin` LCSA threshold path.
- Extended CUDA profiling across LQ transfer/cast, pixel unshuffle, causal
  Conv3d stages, normalization/activation, cache updates, linear projection,
  model work, LCSA routing, Sparge stages, and result assembly.
- Added LQ weight device, dtype, and operation-class diagnostics per streaming
  segment.

For detailed pre-release development history, consult earlier tagged packages.
