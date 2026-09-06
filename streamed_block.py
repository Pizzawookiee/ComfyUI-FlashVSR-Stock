"""FlashVSR-only low-VRAM execution for stock ComfyUI Wan blocks.

This keeps stock Wan math and ordering but shortens sequence-sized temporary
lifetimes. Kitchen gets an additional direct-projection route implemented in
kitchen_projected.py; Sparge and dense attention keep the normal attention
dispatcher while still benefiting from FFN and Linear activation chunking.
"""

from __future__ import annotations

import math
import torch


_MIN_CHUNK = 128
_MAX_CHUNK = 8192
_WORK_BUDGET = 96 * 1024 * 1024


def _align_down(value, alignment):
    value = int(value)
    alignment = max(1, int(alignment))
    return max(alignment, value // alignment * alignment)


def _runtime_from_options(default_runtime, transformer_options):
    options = transformer_options or {}
    return options.get("flashvsr_active_runtime", default_runtime)


def _streaming(runtime):
    return bool(runtime is not None and getattr(runtime, "streaming_active", False))


def _budget(device):
    budget = _WORK_BUDGET
    if device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            budget = min(
                budget,
                max(32 * 1024 * 1024, int(free_bytes * 0.08)),
            )
        except (RuntimeError, TypeError):
            pass
    return int(budget)


def _ffn_chunk_tokens(x, block):
    dim = int(x.shape[-1])
    try:
        ffn_dim = int(block.ffn[0].out_features)
    except (AttributeError, TypeError, IndexError):
        ffn_dim = dim * 4
    element = max(2, int(x.element_size()))
    bytes_per_token = ffn_dim * (element + 1) + dim * element * 4
    tokens = _budget(x.device) // max(1, bytes_per_token)
    return max(
        _MIN_CHUNK,
        min(_MAX_CHUNK, _align_down(max(_MIN_CHUNK, tokens), _MIN_CHUNK)),
    )


def _linear_chunk_tokens(x, module):
    in_dim = int(x.shape[-1])
    out_dim = int(getattr(module, "out_features", in_dim) or in_dim)
    element = max(2, int(x.element_size()))
    bytes_per_token = in_dim * (element + 1) + out_dim * element
    tokens = _budget(x.device) // max(1, bytes_per_token)
    return max(
        _MIN_CHUNK,
        min(_MAX_CHUNK, _align_down(max(_MIN_CHUNK, tokens), _MIN_CHUNK)),
    )


def _modulation_indices(e, total_tokens, token_indices):
    if e.size(1) <= 1:
        return None
    repeats = total_tokens // e.size(1)
    if repeats * e.size(1) != total_tokens:
        repeats += 1
    if repeats <= 0:
        repeats = 1
    mapped = torch.div(token_indices, repeats, rounding_mode="floor")
    return mapped.clamp_max(e.size(1) - 1)


def _gather_e(e, total_tokens, token_indices):
    mapped = _modulation_indices(e, total_tokens, token_indices)
    if mapped is None:
        return e
    return e.index_select(1, mapped)


def _slice_e(e, total_tokens, start, end, device):
    indices = torch.arange(start, end, device=device, dtype=torch.long)
    return _gather_e(e, total_tokens, indices)


def _modulated_gather(norm, x, shift, scale, token_indices):
    total = int(x.shape[1])
    current = x.index_select(1, token_indices).contiguous()
    shift_rows = _gather_e(shift, total, token_indices)
    scale_rows = _gather_e(scale, total, token_indices)
    if getattr(norm, "elementwise_affine", False) is False:
        from .wan_fused_ops import layer_norm_modulate
        fused = layer_norm_modulate(
            current, shift_rows, scale_rows, getattr(norm, "eps", 1e-5)
        )
        if fused is not None:
            return fused
    current = norm(current)
    current.mul_(scale_rows + 1)
    current.add_(shift_rows)
    return current


def _modulated_slice(norm, x, shift, scale, start, end):
    total = int(x.shape[1])
    current = x[:, start:end]
    shift_rows = _slice_e(shift, total, start, end, x.device)
    scale_rows = _slice_e(scale, total, start, end, x.device)
    if getattr(norm, "elementwise_affine", False) is False:
        from .wan_fused_ops import layer_norm_modulate
        fused = layer_norm_modulate(
            current, shift_rows, scale_rows, getattr(norm, "eps", 1e-5)
        )
        if fused is not None:
            return fused
    current = norm(current)
    current.mul_(scale_rows + 1)
    current.add_(shift_rows)
    return current


def _build_modulated_input(norm, x, shift, scale, chunk_tokens):
    output = torch.empty_like(x)
    total = int(x.shape[1])
    for start in range(0, total, chunk_tokens):
        end = min(start + chunk_tokens, total)
        current = _modulated_slice(norm, x, shift, scale, start, end)
        output[:, start:end].copy_(current)
        del current
    return output


def _gate_add_inplace(x, y, gate, chunk_tokens):
    if gate.size(1) == 1:
        from .wan_fused_ops import gate_add_inplace
        if gate_add_inplace(x, y, gate):
            return
    total = int(x.shape[1])
    for start in range(0, total, chunk_tokens):
        end = min(start + chunk_tokens, total)
        gate_rows = _slice_e(gate, total, start, end, x.device)
        x[:, start:end].addcmul_(y[:, start:end], gate_rows)


class _ChunkedLinearForward:
    """Preserve a ComfyUI Linear's implementation, but feed token slabs."""

    def __init__(self, module, original_forward, runtime):
        self.module = module
        self.original_forward = original_forward
        self.runtime = runtime

    def __call__(self, x, *args, **kwargs):
        if (
            not _streaming(self.runtime)
            or not torch.is_tensor(x)
            or x.ndim != 3
        ):
            return self.original_forward(x, *args, **kwargs)
        chunk = _linear_chunk_tokens(x, self.module)
        if int(x.shape[1]) <= chunk:
            return self.original_forward(x, *args, **kwargs)
        output = None
        total = int(x.shape[1])
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            current = self.original_forward(x[:, start:end], *args, **kwargs)
            if output is None:
                output = current.new_empty((*x.shape[:-1], current.shape[-1]))
            output[:, start:end].copy_(current)
            del current
        return output


class _WanRopeEncodeCache:
    """Cache stock Wan's spatial RoPE rotations across FlashVSR chunks.

    FlashVSR changes only the temporal shift between internal model calls.
    Height/width positions and their rotations are invariant at a fixed video
    geometry, so retain only that compact spatial component and rebuild the
    shifted temporal component for each chunk. Unsupported/non-streaming calls
    use stock ComfyUI's bound rope_encode unchanged.
    """

    MAX_SPATIAL_ENTRIES = 4

    def __init__(self, module, original_forward, runtime):
        self.module = module
        self.original_forward = original_forward
        self.runtime = runtime
        self.spatial_cache = {}

    def __call__(
        self,
        t,
        h,
        w,
        t_start=0,
        steps_t=None,
        steps_h=None,
        steps_w=None,
        device=None,
        dtype=None,
        transformer_options=None,
        source_id=0,
        **kwargs,
    ):
        options = transformer_options or {}
        runtime = _runtime_from_options(self.runtime, options)
        embedder = getattr(self.module, "rope_embedder", None)
        axes_dim = getattr(embedder, "axes_dim", None)
        theta = getattr(embedder, "theta", None)
        patch_size = getattr(self.module, "patch_size", None)
        if (
            not _streaming(runtime)
            or source_id
            or kwargs
            or device is None
            or dtype is None
            or patch_size is None
            or axes_dim is None
            or len(axes_dim) != 3
            or theta is None
        ):
            return self.original_forward(
                t, h, w,
                t_start=t_start,
                steps_t=steps_t,
                steps_h=steps_h,
                steps_w=steps_w,
                device=device,
                dtype=dtype,
                transformer_options=options,
                source_id=source_id,
                **kwargs,
            )

        # Match stock comfy.ldm.wan.model.WanModel.rope_encode exactly.
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        if steps_t is None:
            steps_t = t_len
        if steps_h is None:
            steps_h = h_len
        if steps_w is None:
            steps_w = w_len

        h_start = 0
        w_start = 0
        rope_options = options.get("rope_options")
        if rope_options is not None:
            t_len = (t_len - 1.0) * rope_options.get("scale_t", 1.0) + 1.0
            h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
            w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0
            t_start += rope_options.get("shift_t", 0.0)
            h_start += rope_options.get("shift_y", 0.0)
            w_start += rope_options.get("shift_x", 0.0)

        steps_t = int(steps_t)
        steps_h = int(steps_h)
        steps_w = int(steps_w)
        target = torch.device(device)
        cache_key = (
            steps_h, steps_w,
            float(h_start), float(h_len),
            float(w_start), float(w_len),
            int(axes_dim[1]), int(axes_dim[2]), float(theta),
            dtype, target.type, target.index,
        )
        spatial = self.spatial_cache.get(cache_key)
        if spatial is None:
            from comfy.ldm.flux.math import rope

            h_pos = torch.linspace(
                h_start, h_start + (h_len - 1),
                steps=steps_h, device=target, dtype=dtype,
            )
            w_pos = torch.linspace(
                w_start, w_start + (w_len - 1),
                steps=steps_w, device=target, dtype=dtype,
            )
            h_rot = rope(h_pos.unsqueeze(0), int(axes_dim[1]), theta)[0]
            w_rot = rope(w_pos.unsqueeze(0), int(axes_dim[2]), theta)[0]
            h_rot = h_rot[:, None].expand(
                steps_h, steps_w, *h_rot.shape[1:]
            )
            w_rot = w_rot[None, :].expand(
                steps_h, steps_w, *w_rot.shape[1:]
            )
            spatial = torch.cat((h_rot, w_rot), dim=-3).reshape(
                steps_h * steps_w, -1, 2, 2
            ).contiguous()
            if len(self.spatial_cache) >= self.MAX_SPATIAL_ENTRIES:
                self.spatial_cache.clear()
            self.spatial_cache[cache_key] = spatial

        from comfy.ldm.flux.math import rope

        t_pos = torch.linspace(
            t_start, t_start + (t_len - 1),
            steps=steps_t, device=target, dtype=dtype,
        )
        temporal = rope(t_pos.unsqueeze(0), int(axes_dim[0]), theta)[0]
        spatial_tokens = steps_h * steps_w
        temporal_pairs = temporal.shape[-3]
        spatial_pairs = spatial.shape[-3]
        freqs = torch.empty(
            (steps_t, spatial_tokens, temporal_pairs + spatial_pairs, 2, 2),
            device=target, dtype=temporal.dtype,
        )
        freqs[:, :, :temporal_pairs].copy_(temporal[:, None])
        freqs[:, :, temporal_pairs:].copy_(spatial[None])
        return freqs.reshape(
            1, steps_t * spatial_tokens, 1,
            temporal_pairs + spatial_pairs, 2, 2,
        )


class _SteadyThresholdMask:
    """PR #108-style per-block steady LCSA threshold reuse.

    Prefill and the first continuation use the original top-k threshold. Later
    continuations reuse that first steady threshold for the same block/geometry.
    The best-key scatter remains active, so every query row always has a route.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.thresholds = {}

    def __call__(self, q_pool, k_pool, grid_h, grid_w):
        runtime = self.runtime
        batch, q_temporal, spatial, heads, head_dim = q_pool.shape
        k_temporal = k_pool.shape[1]
        if (
            k_pool.shape[0] != batch
            or k_pool.shape[2] != spatial
            or k_pool.shape[3] != heads
            or k_pool.shape[4] != head_dim
        ):
            raise RuntimeError("Invalid FlashVSR cached LCSA K summary.")
        scores = torch.einsum(
            "btnhd,bsmhd->bhtnsm", q_pool, k_pool
        ) / math.sqrt(head_dim)
        scores = scores.reshape(
            batch, heads, q_temporal, spatial,
            k_temporal * spatial,
        ).float()

        local, eligible_count = runtime._local_block_topology(
            q_temporal, k_temporal, grid_h, grid_w,
            runtime.lcsa_local_range, scores.device,
        )
        scores.masked_fill_(~local, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        flat = probabilities.reshape(batch, heads, q_temporal, -1)
        requested = int(spatial * spatial * runtime.lcsa_sparse_ratio) - 1
        selected_count = min(max(1, requested), flat.shape[-1] - 1)
        if selected_count >= eligible_count:
            selected = probabilities > 0
            best = probabilities.argmax(dim=-1, keepdim=True)
            selected.scatter_(-1, best, True)
            return selected

        block_index = int(getattr(runtime, "_flashvsr_threshold_block", -1))
        steady = int(getattr(runtime, "current_rope_start", 0)) > 0
        if not steady and block_index == 0 and self.thresholds:
            self.thresholds.clear()
        key = (
            block_index, batch, heads, q_temporal, grid_h, grid_w,
            k_temporal, runtime.lcsa_local_range, selected_count,
            float(runtime.lcsa_sparse_ratio),
            scores.device.type, scores.device.index,
        )
        threshold = self.thresholds.get(key) if steady and block_index >= 0 else None
        if threshold is None:
            top_values = torch.topk(
                flat, k=selected_count + 1, dim=-1, sorted=False
            ).values
            threshold = top_values.amin(dim=-1, keepdim=True)
            if steady and block_index >= 0:
                self.thresholds[key] = threshold.detach().clone()
        selected = (flat > threshold).view_as(probabilities)
        best = probabilities.argmax(dim=-1, keepdim=True)
        selected.scatter_(-1, best, True)
        return selected


def _install_threshold_cache(runtime):
    wrapper = getattr(runtime, "_flashvsr_threshold_wrapper", None)
    if wrapper is None:
        wrapper = _SteadyThresholdMask(runtime)
        runtime._flashvsr_threshold_wrapper = wrapper
        runtime._lcsa_mask_from_pools = wrapper
    return wrapper


class WanStreamedBlockForward:
    """Memory-bounded equivalent of stock WanAttentionBlock.forward."""

    def __init__(self, module, original_forward, runtime):
        self.module = module
        self.original_forward = original_forward
        self.runtime = runtime
        self._modulation_cache = None
        self._modulation_cache_key = None

    def __call__(
        self,
        x,
        e,
        freqs,
        context,
        context_img_len=257,
        transformer_options=None,
    ):
        options = transformer_options or {}
        runtime = _runtime_from_options(self.runtime, options)
        if not _streaming(runtime):
            return self.original_forward(
                x,
                e,
                freqs,
                context,
                context_img_len=context_img_len,
                transformer_options=options,
            )

        import comfy.model_management

        # FlashVSR is a one-sigma sampler: timestep modulation is invariant
        # across its internal streaming chunks. Cache the already-cast
        # modulation+e decomposition per block and reuse it until the next
        # six-frame prefill (current_rope_start==0) or an input shape changes.
        # This mirrors PR #108's lossless step-invariant modulation cache while
        # keeping full_video_dense and non-FlashVSR model calls untouched.
        cache_key = (
            id(runtime),
            int(e.ndim),
            tuple(int(value) for value in e.shape),
            x.dtype,
            x.device.type,
            x.device.index,
        )
        reset_cache = getattr(runtime, "current_rope_start", 0) == 0
        if reset_cache or self._modulation_cache_key != cache_key:
            modulation = comfy.model_management.cast_to(
                self.module.modulation, dtype=x.dtype, device=x.device
            )
            if e.ndim < 4:
                cached = (modulation + e).chunk(6, dim=1)
            else:
                cached = (modulation.unsqueeze(0) + e).unbind(2)
            self._modulation_cache = tuple(cached)
            self._modulation_cache_key = cache_key
        e = self._modulation_cache

        patches = options.get("patches", {})
        runtime._flashvsr_threshold_block = int(options.get("block_index", -1))
        x = x.contiguous()
        chunk = _ffn_chunk_tokens(x, self.module)

        # Kitchen can consume projected Q/K/V directly from bounded modulated
        # input slabs. This avoids even the one full self-attention input tensor.
        y = None
        backend = getattr(runtime, "sparse_attention_backend", None)
        is_kitchen = (
            backend is not None
            and backend.__class__.__module__.endswith("kitchen_sparse_backend")
            and not patches.get("attn1_patch")
        )
        if is_kitchen:
            from .kitchen_projected import run_projected_kitchen_attention

            def gather_input(indices):
                return _modulated_gather(
                    self.module.norm1, x, e[0], e[1], indices
                )

            y = run_projected_kitchen_attention(
                backend,
                self.module.self_attn,
                x,
                freqs,
                runtime,
                options,
                gather_input,
            )

        if y is None:
            attn_input = _build_modulated_input(
                self.module.norm1, x, e[0], e[1], chunk
            )
            y = self.module.self_attn(
                attn_input,
                freqs,
                transformer_options=options,
            )
            del attn_input

        _gate_add_inplace(x, y, e[2], chunk)
        del y

        norm3 = self.module.norm3(x)
        cross = self.module.cross_attn(
            norm3,
            context,
            context_img_len=context_img_len,
            transformer_options=options,
        )
        if norm3 is not x:
            del norm3
        x.add_(cross)
        del cross

        for patch in patches.get("attn2_patch", ()):
            x = patch({"x": x, "transformer_options": options})

        # FFN: no complete norm2, modulation, expanded activation, INT8 input
        # carrier, or gated output exists for the complete video sequence.
        total = int(x.shape[1])
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            current = _modulated_slice(
                self.module.norm2, x, e[3], e[4], start, end
            )
            y_chunk = self.module.ffn(current)
            del current
            gate = _slice_e(e[5], total, start, end, x.device)
            x[:, start:end].addcmul_(y_chunk, gate)
            del y_chunk
        return x


def _patch_linear(patcher, path, module, runtime):
    patcher.add_object_patch(
        f"{path}.forward",
        _ChunkedLinearForward(module, module.forward, runtime),
    )


def install_streamed_wan_block_patches(patcher, runtime, blocks):
    if not blocks:
        return
    _install_threshold_cache(runtime)

    # Stock Wan reconstructs the complete T/H/W RoPE tensor every model call.
    # FlashVSR varies only temporal shift across its streaming chunks, so patch
    # the model-level encoder to reuse the invariant spatial rotations.
    diffusion_model = None
    try:
        diffusion_model = patcher.get_model_object("diffusion_model")
    except (AttributeError, KeyError, TypeError, RuntimeError):
        diffusion_model = getattr(
            getattr(patcher, "model", None), "diffusion_model", None
        )
    rope_encode = getattr(diffusion_model, "rope_encode", None)
    module_name = getattr(type(diffusion_model), "__module__", "")
    if callable(rope_encode) and module_name == "comfy.ldm.wan.model":
        patcher.add_object_patch(
            "diffusion_model.rope_encode",
            _WanRopeEncodeCache(diffusion_model, rope_encode, runtime),
        )

    for index, block in enumerate(blocks):
        required = (
            "norm1", "self_attn", "norm2", "norm3",
            "cross_attn", "ffn", "modulation",
        )
        if not all(hasattr(block, name) for name in required):
            continue
        base = f"diffusion_model.blocks.{index}"
        patcher.add_object_patch(
            f"{base}.forward",
            WanStreamedBlockForward(block, block.forward, runtime),
        )
        for attention_name in ("self_attn", "cross_attn"):
            attention = getattr(block, attention_name, None)
            if attention is None:
                continue
            for linear_name in ("q", "k", "v", "o", "k_img", "v_img"):
                linear = getattr(attention, linear_name, None)
                if linear is not None and hasattr(linear, "forward"):
                    _patch_linear(
                        patcher,
                        f"{base}.{attention_name}.{linear_name}",
                        linear,
                        runtime,
                    )
