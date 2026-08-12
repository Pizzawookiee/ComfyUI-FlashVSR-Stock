from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management as model_management
import comfy.model_sampling
import comfy.sd
import comfy.utils

from .components import load_lq_projector, load_tcdecoder
from .color import apply_color_correction
from .model_paths import component_filenames, full_path
from .runtime import FlashVSRRuntime, patch_model, prepare_video
from .sampler import SAMPLING_MODES, make_sampler
from .sparse_backend import apply_sparge_backend


DTYPES = ["auto", "bf16", "fp16", "fp32"]


class FlashVSRModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (component_filenames("dit"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "FlashVSR/loaders"

    def load(self, model_name):
        model = comfy.sd.load_diffusion_model(full_path(model_name))
        patched = model.clone()
        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class FlashVSRModelSampling(sampling_base, sampling_type):
            pass

        original = patched.get_model_object("model_sampling")
        sampling = FlashVSRModelSampling(model.model.model_config)
        sampling.set_parameters(shift=5.0, multiplier=1000)
        if hasattr(original, "noise_scale"):
            sampling.set_noise_scale(original.noise_scale)
        patched.add_object_patch("model_sampling", sampling)
        return (patched,)


class FlashVSRPromptLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt_name": (component_filenames("prompt"),)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "FlashVSR/loaders"

    def load(self, prompt_name):
        sd = comfy.utils.load_torch_file(full_path(prompt_name), safe_load=True)
        if "context" in sd:
            context = sd["context"]
        elif len(sd) == 1:
            context = next(iter(sd.values()))
        else:
            raise ValueError("Prompt safetensors must contain one tensor or a 'context' tensor.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        return ([[context.cpu(), {}]],)


class FlashVSRLQLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lq_name": (component_filenames("lq"),),
            "compute_dtype": (DTYPES, {
                "tooltip": (
                    "LQ projector compute and stored floating-weight dtype. "
                    "FP16 or BF16 converts the released FP32 weights once "
                    "while loading, reducing repeated cast work and managed "
                    "weight memory. FP32 preserves released storage exactly."
                ),
            }),
        }}

    RETURN_TYPES = ("FLASHVSR_LQ",)
    FUNCTION = "load"
    CATEGORY = "FlashVSR/loaders"

    def load(self, lq_name, compute_dtype):
        return (load_lq_projector(full_path(lq_name), compute_dtype),)


class FlashVSRTCDecoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "decoder_name": (component_filenames("decoder"),),
            "compute_dtype": (DTYPES,),
        }, "optional": {
            "fuse_tgrow": ("BOOLEAN", {
                "default": False,
                "advanced": True,
                "tooltip": (
                    "Composes each linear temporal-growth 1x1 convolution "
                    "with its following 3x3 convolution once while loading. "
                    "This removes three decoder convolutions and their "
                    "intermediate activations. It is mathematically "
                    "equivalent, but may select a larger cuDNN workspace. "
                    "Leave disabled for the low-VRAM baseline."
                ),
            }),
            "channels_last": ("BOOLEAN", {
                "default": False,
                "advanced": True,
                "tooltip": (
                    "Uses channels-last Conv2d weights and activations. This "
                    "can improve throughput on some GPUs, but cuDNN may pick "
                    "algorithms with larger temporary workspaces. Leave "
                    "disabled for the low-VRAM baseline."
                ),
            }),
        }}

    RETURN_TYPES = ("FLASHVSR_DECODER",)
    FUNCTION = "load"
    CATEGORY = "FlashVSR/loaders"

    def load(self, decoder_name, compute_dtype, fuse_tgrow=False,
             channels_last=False):
        return (
            load_tcdecoder(
                full_path(decoder_name),
                compute_dtype,
                fuse_tgrow=fuse_tgrow,
                channels_last=channels_last,
            ),
        )


class FlashVSRPrepareVideo:
    DESCRIPTION = (
        "Packs frames that have already been resized to the desired final "
        "resolution. This node does not upscale. It adds only FlashVSR's "
        "spatial and temporal padding and creates the matching Wan latent."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {
                "tooltip": (
                    "Frames already resized to the intended output width "
                    "and height. Use stock ImageScale/ImageScaleBy or any "
                    "other resizing node first. FlashVSR is intended for "
                    "4x restoration, although other scales may run."
                ),
            }),
        }}

    RETURN_TYPES = ("FLASHVSR_VIDEO", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("video", "latent", "width", "height", "output_frames")
    FUNCTION = "prepare"
    CATEGORY = "FlashVSR"

    def prepare(self, images):
        prepared = prepare_video(images)
        return (
            prepared,
            prepared.latent,
            prepared.output_width,
            prepared.output_height,
            prepared.original_frames,
        )


class FlashVSRApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lq_projector": ("FLASHVSR_LQ",),
            "video": ("FLASHVSR_VIDEO",),
            "conditioning_strength": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": (
                    "Multiplier for resized-video conditioning injected at "
                    "Wan block 0. 1.0 matches the checkpoint. Lower values "
                    "give the model more freedom; higher values follow the "
                    "input more strongly and may preserve its artifacts."
                ),
            }),
        }}

    RETURN_TYPES = ("MODEL", "FLASHVSR_RUNTIME")
    FUNCTION = "apply"
    CATEGORY = "FlashVSR"

    def apply(self, model, lq_projector, video, conditioning_strength):
        runtime = FlashVSRRuntime(
            lq_projector, video, conditioning_strength
        )
        return patch_model(model, runtime), runtime


class FlashVSRBlockSparseAttention:
    DESCRIPTION = (
        "Uses the separately installed SpargeAttn library to execute "
        "FlashVSR's native 128-query by 64-key LCSA mask during streaming. "
        "It does not replace ModelAttentionBackend: dense self-attention, "
        "cross-attention, and full_video_dense continue through the model's "
        "selected ComfyUI backend. Patch order relative to Configure "
        "FlashVSR Upscaling and ModelAttentionBackend does not matter. A "
        "matching SpargeAttn wheel and CUDA GPU are required."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FlashVSR/model_patches"

    def patch(self, model):
        return (apply_sparge_backend(model),)


class FlashVSRStreamingSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "runtime": ("FLASHVSR_RUNTIME",),
            "sampling_mode": (SAMPLING_MODES, {
                "default": "full_video_dense",
                "tooltip": (
                    "full_video_dense processes the complete clip in one "
                    "dense-attention model call: useful as a reference but "
                    "VRAM grows sharply with resolution and duration; the "
                    "LCSA controls below are ignored. streaming processes "
                    "bounded overlapping segments and applies FlashVSR LCSA "
                    "through the model's selected mask-capable attention "
                    "backend, greatly reducing long-video VRAM. The optional "
                    "FlashVSR Sparge Attention additionally skips rejected "
                    "key blocks with SpargeAttn instead of applying the "
                    "sparse topology through a dense masked kernel."
                ),
            }),
            "sparse_ratio": ("FLOAT", {
                "default": 2.0,
                "min": 1.0,
                "max": 4.0,
                "step": 0.1,
                "advanced": True,
                "tooltip": (
                    "Streaming-only LCSA top-k block-pair budget. Higher "
                    "values retain more attention connections and usually "
                    "improve motion/detail stability; lower values are more "
                    "aggressive and may look sharper but can flicker or lose "
                    "structure. 2.0 is the stable default. With the current "
                    "dense mask backend this is mainly a quality control, "
                    "not a major speed or VRAM control."
                ),
            }),
            "local_range": ("INT", {
                "default": 11,
                "min": 3,
                "max": 15,
                "step": 2,
                "advanced": True,
                "tooltip": (
                    "Streaming-only spatial search neighborhood measured in "
                    "8x8 Wan-token windows (about 128 output pixels per "
                    "window). Larger values better accommodate fast or large "
                    "motion; smaller values keep attention more local and may "
                    "look sharper but can destabilize moving objects. 11 is "
                    "the stable default."
                ),
            }),
            "query_block_chunk": ("INT", {
                "default": 1,
                "min": 0,
                "max": 32,
                "step": 1,
                "advanced": True,
                "tooltip": (
                    "Number of 128-query LCSA windows sent to the selected "
                    "backend at once. 0 selects a conservative value from "
                    "currently free VRAM; 1 minimizes mask VRAM; larger "
                    "manual values are faster but use more VRAM. This control "
                    "is ignored by FlashVSR Sparge Attention, which processes "
                    "the complete compact block mask in one sparse call."
                ),
            }),
        }, "optional": {
            "new_latent_frames": ("INT", {
                "default": 4,
                "min": 2,
                "max": 4,
                "step": 2,
                "advanced": True,
                "tooltip": (
                    "New latent frames appended per continuation. 4 is the "
                    "optimized default: it uses six-frame continuations, "
                    "reducing calls without exceeding the existing six-frame "
                    "initial peak. Use 2 as the compatibility control."
                ),
            }),
            "profile_cuda_events": ("BOOLEAN", {
                "default": False,
                "advanced": True,
                "tooltip": (
                    "Print synchronized CUDA-event timings after sampling "
                    "for LQ transfers, pixel unshuffle, Conv3d, norm/SiLU, "
                    "cache updates and linear projection; complete Wan "
                    "calls; LCSA routing; Sparge layout/mask/kernel/restore; "
                    "and result assembly. It also prints LQ weight residency "
                    "and verifies BasicGuider/CFG=1 pass count. Off adds no "
                    "CUDA events; on is intended for benchmarking."
                ),
            }),
        }}

    RETURN_TYPES = ("SAMPLER", "SIGMAS")
    FUNCTION = "build"
    CATEGORY = "FlashVSR/sampling"

    def build(self, runtime, sampling_mode, sparse_ratio, local_range,
              query_block_chunk, new_latent_frames=4,
              profile_cuda_events=False):
        return (
            make_sampler(
                runtime, sampling_mode, sparse_ratio,
                local_range, query_block_chunk, new_latent_frames,
                profile_cuda_events,
            ),
            torch.tensor([1.0, 0.0], dtype=torch.float32),
        )


class FlashVSRTCDecode:
    DESCRIPTION = (
        "Decodes TCDecoder output with four-frame conditioning streamed from "
        "CPU, reusable temporal-state buffers, native compute-dtype weights, "
        "a true frame-sequential low-VRAM path, and bounded temporal/output "
        "batches. Use temporal_batch_size=1 and output_chunk_size=1 for "
        "minimum VRAM."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "decoder": ("FLASHVSR_DECODER",),
                "video": ("FLASHVSR_VIDEO",),
            },
            "optional": {
                "output_chunk_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "advanced": True,
                    "tooltip": (
                        "Decoded frames staged on the GPU before one copy "
                        "to the final IMAGE tensor. 1 minimizes VRAM; 4 "
                        "matches TCDecoder's temporal output group."
                    ),
                }),
                "temporal_batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "advanced": True,
                    "tooltip": (
                        "Latent timesteps decoded together. 1 uses a true "
                        "depth-first path with one generated high-resolution "
                        "frame active at a time; 2 often improves GPU "
                        "utilization substantially; 4 is faster when it fits "
                        "but can use roughly 1.5-3x the decoder activation "
                        "VRAM. This remains bounded and does not grow with "
                        "the complete clip duration."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "FlashVSR/decoding"

    def decode(self, samples, decoder, video, output_chunk_size=4,
               temporal_batch_size=1):
        model_management.load_models_gpu([decoder.patcher])
        device = decoder.patcher.load_device
        dtype = decoder.compute_dtype
        latent_format = comfy.latent_formats.Wan21()
        latents = samples["samples"]
        condition = video.tensor[:, :, :video.generated_frames]
        decoder.model.clean_mem()
        try:
            # TCDecoder is trained to emit display-range RGB. The official
            # wrapper's temporary [-1, 1] conversion is undone by its video
            # postprocessor; ComfyUI IMAGE uses [0, 1] directly.
            frames = decoder.model.decode_video(
                latents,
                condition,
                compute_device=device,
                compute_dtype=dtype,
                latent_mean=latent_format.latents_mean,
                latent_std=latent_format.latents_std,
                latent_scale_factor=latent_format.scale_factor,
                output_device=model_management.intermediate_device(),
                output_dtype=torch.float32,
                output_chunk_size=output_chunk_size,
                temporal_batch_size=temporal_batch_size,
                frame_start=video.crop_start,
                frame_count=video.original_frames,
                output_height=video.output_height,
                output_width=video.output_width,
                clamp_output=True,
            )
            images = frames[0].movedim(1, -1)
            return (images,)
        finally:
            decoder.model.clean_mem()


class FlashVSRCropFrames:
    DESCRIPTION = (
        "Postprocesses stock Wan VAE output by trimming tail-padding frames "
        "to the original frame count and removing the right/bottom padding "
        "added for 128-pixel spatial alignment. FlashVSR's official causal "
        "layout starts directly at source frame zero. Fast AdaIN and "
        "low-frequency wavelet color correction are optional."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {
                "tooltip": (
                    "Decoded padded frames from stock VAE Decode. The node "
                    "removes duplicated tail frames and right/bottom spatial "
                    "alignment padding."
                ),
            }),
            "video": ("FLASHVSR_VIDEO", {
                "tooltip": (
                    "Prepared-video metadata used to restore the requested "
                    "frame count, width, and height."
                ),
            }),
            "color_correction": ([
                "off",
                "adain",
                "wavelet_quarter_res",
                "wavelet_full_res",
            ], {
                "default": "off",
                "tooltip": (
                    "off preserves the VAE output. adain is the fast option: "
                    "it matches each frame's per-channel mean and contrast "
                    "to the resized input. wavelet_quarter_res transfers "
                    "spatially varying low-frequency color at quarter width "
                    "and height and is the practical wavelet option. "
                    "wavelet_full_res performs all five blur passes at the "
                    "complete output resolution for an exact but much "
                    "slower result."
                ),
            }),
            "color_chunk_size": ("INT", {
                "default": 4,
                "min": 1,
                "max": 64,
                "step": 1,
                "advanced": True,
                "tooltip": (
                    "Frames color-corrected together. Larger batches reduce "
                    "per-frame overhead; 1 minimizes temporary memory. This "
                    "also batches AdaIN when that mode is selected."
                ),
            }),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "FlashVSR/decoding"

    def crop(self, images, video, color_correction="off",
             color_chunk_size=4):
        images = images[video.crop_start:video.crop_start + video.original_frames]
        images = images[:, :video.output_height, :video.output_width]
        if color_correction != "off":
            images = apply_color_correction(
                images,
                video,
                method=color_correction,
                chunk_size=color_chunk_size,
            )
        return (images,)


NODE_CLASS_MAPPINGS = {
    "FlashVSRModelLoader": FlashVSRModelLoader,
    "FlashVSRPromptLoader": FlashVSRPromptLoader,
    "FlashVSRLQLoader": FlashVSRLQLoader,
    "FlashVSRTCDecoderLoader": FlashVSRTCDecoderLoader,
    "FlashVSRPrepareVideo": FlashVSRPrepareVideo,
    "FlashVSRApply": FlashVSRApply,
    "FlashVSRBlockSparseAttention": FlashVSRBlockSparseAttention,
    "FlashVSRStreamingSampler": FlashVSRStreamingSampler,
    "FlashVSRTCDecode": FlashVSRTCDecode,
    "FlashVSRCropFrames": FlashVSRCropFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRModelLoader": "Load FlashVSR DiT (Stock Wan)",
    "FlashVSRPromptLoader": "Load FlashVSR Prompt",
    "FlashVSRLQLoader": "Load FlashVSR LQ Projector",
    "FlashVSRTCDecoderLoader": "Load FlashVSR TCDecoder",
    "FlashVSRPrepareVideo": "Prepare Video for FlashVSR",
    "FlashVSRApply": "Configure FlashVSR Upscaling",
    "FlashVSRBlockSparseAttention": (
        "FlashVSR Sparge Attention"
    ),
    "FlashVSRStreamingSampler": "FlashVSR One-Step Sampler",
    "FlashVSRTCDecode": "FlashVSR Tiny Decode",
    "FlashVSRCropFrames": "FlashVSR Postprocess",
}
