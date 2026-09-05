"""FlashVSR-only low-VRAM execution for stock ComfyUI Wan blocks.

This keeps stock Wan math and ordering but shortens sequence-sized temporary
lifetimes. Kitchen gets an additional direct-projection route implemented in
kitchen_projected.py; Sparge and dense attention keep the normal attention
dispatcher while still benefiting from FFN and Linear activation chunking.
"""

from __future__ import annotations

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
    current = x.index_select(1, token_indices)
    current = norm(current)
    shift_rows = _gather_e(shift, total, token_indices)
    scale_rows = _gather_e(scale, total, token_indices)
    current.mul_(scale_rows + 1)
    current.add_(shift_rows)
    return current


def _modulated_slice(norm, x, shift, scale, start, end):
    total = int(x.shape[1])
    current = norm(x[:, start:end])
    shift_rows = _slice_e(shift, total, start, end, x.device)
    scale_rows = _slice_e(scale, total, start, end, x.device)
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
