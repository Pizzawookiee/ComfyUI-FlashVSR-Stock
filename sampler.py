from __future__ import annotations

import math

import torch

import comfy.samplers
import comfy.utils

from .runtime import ACTIVE_RUNTIME_OPTION, FlashVSRRuntime


SAMPLING_MODES = [
    "full_video_dense",
    "streaming",
]


def make_sampler(runtime: FlashVSRRuntime, sampling_mode: str,
                 sparse_ratio: float = 2.0,
                 local_range: int = 11,
                 query_block_chunk: int = 1,
                 new_latent_frames: int = 4,
                 profile_cuda_events: bool = False):
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown FlashVSR sampling mode: {sampling_mode}")
    configured_new_latent_frames = int(new_latent_frames)
    if configured_new_latent_frames not in (2, 4):
        raise ValueError(
            "FlashVSR new_latent_frames must be 2 or 4."
        )

    def flashvsr_sample(model, noise, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        if len(sigmas) < 2:
            return noise
        extra_args = dict(extra_args or {})
        model_options = dict(extra_args.get("model_options") or {})
        transformer_options = dict(
            model_options.get("transformer_options") or {}
        )
        model_options["transformer_options"] = transformer_options
        extra_args["model_options"] = model_options
        # MODEL and SAMPLER outputs can be cached independently on warm runs.
        # Bind all model-resident FlashVSR callbacks to the runtime that is
        # preparing tokens for this exact sampler invocation.
        transformer_options[ACTIVE_RUNTIME_OPTION] = runtime

        runtime.reset()
        runtime.begin_profile(profile_cuda_events)
        sampling_marker = runtime.profile_start(noise)
        if profile_cuda_events:
            guider = getattr(model, "inner_model", None)
            guider_name = (
                type(guider).__name__ if guider is not None else "unknown"
            )
            cfg = getattr(guider, "cfg", None)
            conds = getattr(guider, "conds", {}) or {}
            negative_present = conds.get("negative") is not None
            cfg1_disabled = bool(
                model_options.get("disable_cfg1_optimization", False)
            )
            if not negative_present:
                state = "single conditional pass (no negative conditioning)"
            elif (
                cfg is not None
                and math.isclose(float(cfg), 1.0)
                and not cfg1_disabled
            ):
                state = "single conditional pass (CFG=1 optimization active)"
            elif cfg is None:
                state = "could not verify pass count"
            else:
                state = "conditional + negative guidance work is active"
            cfg_text = "unknown" if cfg is None else f"{float(cfg):g}"
            print(
                f"[FlashVSR profiler] guider={guider_name}, cfg={cfg_text}, "
                f"disable_cfg1_optimization={cfg1_disabled}: {state}."
            )
        try:
            if sampling_mode == "full_video_dense":
                runtime.begin_sampling(streaming=False)
                transformer_options.pop("rope_options", None)
                lq_marker = runtime.profile_start(noise)
                runtime.prepare_lq_full(noise.shape[2])
                runtime.profile_end("lq_projector", lq_marker)
                sigma = sigmas[0].expand(noise.shape[0])
                model_marker = runtime.profile_start(noise)
                denoised = model(noise, sigma, **extra_args)
                runtime.profile_end("model_total", model_marker)
                if callback is not None:
                    callback({
                        "x": noise,
                        "i": 0,
                        "sigma": sigmas[0],
                        "sigma_hat": sigmas[0],
                        "denoised": denoised,
                    })
                return denoised

            latent_frames = noise.shape[2]
            if latent_frames < 6 or latent_frames % 2:
                raise ValueError(
                    "FlashVSR streaming requires an even latent length "
                    "of at least six."
                )
            segments = [(0, 0, 6)]
            output_start = 6
            process_index = 1
            while output_start < latent_frames:
                new_count = min(
                    configured_new_latent_frames,
                    latent_frames - output_start,
                )
                segments.append((process_index, output_start, new_count))
                output_start += new_count
                process_index += new_count // 2

            runtime.begin_sampling(
                streaming=True,
                sparse_ratio=sparse_ratio,
                local_range=local_range,
                query_block_chunk=query_block_chunk,
            )
            runtime.force_streaming_attention_override(transformer_options)
            progress = comfy.utils.ProgressBar(len(segments) + 1)
            result = None
            write_start = 0
            for segment_number, (
                process_index, output_start, new_count
            ) in enumerate(segments):
                lq_marker = runtime.profile_start(noise)
                runtime.prepare_lq_for_process(
                    process_index, new_count
                )
                runtime.profile_end("lq_projector", lq_marker)
                runtime.begin_model_chunk()
                transformer_options["rope_options"] = {
                    "shift_t": runtime.current_rope_start,
                }
                if segment_number == 0:
                    current = noise[:, :, :6]
                else:
                    # Re-run the previous two latent frames beside two or four
                    # new frames. Four-new mode keeps the existing six-frame
                    # peak while reducing continuation calls and boundaries.
                    current = noise[
                        :, :, output_start - 2:output_start + new_count
                    ]

                sigma = sigmas[0].expand(current.shape[0])
                model_marker = runtime.profile_start(current)
                denoised = model(current, sigma, **extra_args)
                runtime.profile_end("model_total", model_marker)
                assembly_marker = runtime.profile_start(denoised)
                selected = (
                    denoised
                    if segment_number == 0
                    else denoised[:, :, -new_count:]
                )
                if result is None:
                    result = denoised.new_empty(noise.shape)
                write_end = write_start + selected.shape[2]
                result[:, :, write_start:write_end].copy_(selected)
                write_start = write_end
                runtime.profile_end("result_assembly", assembly_marker)
                del selected, denoised
                progress.update(1)

            if result is None or write_start != latent_frames:
                raise RuntimeError(
                    "FlashVSR streaming produced an incomplete latent: "
                    f"{write_start} of {latent_frames} frames."
                )
            # ComfyUI sees FlashVSR as one sigma step. Calling its sampler
            # callback for every internal streaming segment made the standard
            # progress bar reach 100% after the first segment and repeatedly
            # generated previews. Report the one actual step only after the
            # complete latent has been assembled.
            if callback is not None:
                callback({
                    "x": noise,
                    "i": 0,
                    "sigma": sigmas[0],
                    "sigma_hat": sigmas[0],
                    "denoised": result,
                })
            progress.update(1)
            return result
        finally:
            try:
                runtime.profile_end("sampling_total", sampling_marker)
                runtime.finish_profile()
            finally:
                # Profiling failures must not retain LQ state or attention
                # routing across cached warm runs.
                runtime.cleanup()

    return comfy.samplers.KSAMPLER(flashvsr_sample)
