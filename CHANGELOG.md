# Changelog

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
