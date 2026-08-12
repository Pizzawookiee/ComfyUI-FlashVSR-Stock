# Changelog

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
