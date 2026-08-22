"""Bounded asynchronous CPU write-through for mutable FlashVSR carriers.

The CPU copy remains authoritative so AIMDO may evict its opportunistic GPU
mirror at any time.  Only two pinned staging pairs are retained; completed
staging data is copied into ordinary CPU tensors by background workers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import torch

from .qkv import Int8Carrier


def _components(value):
    if isinstance(value, Int8Carrier):
        return (value.qdata, value.scale)
    if isinstance(value, torch.Tensor):
        return (value,)
    raise TypeError(f"Unsupported FlashVSR cache value: {type(value)!r}")


def _from_components(template, components):
    if isinstance(template, Int8Carrier):
        return Int8Carrier(
            components[0], components[1], template.shape,
            template.dtype, template.head_dim,
        )
    return components[0].view(template.shape)


def _empty_cpu_like(value):
    # ComfyUI executes nodes under inference_mode. Tensors allocated there
    # cannot be mutated by the background writer because inference mode is
    # thread-local. CPU-authoritative cache values must therefore be ordinary
    # mutable tensors before they are handed to a worker thread.
    with torch.inference_mode(False):
        components = tuple(
            torch.empty(component.shape, device="cpu", dtype=component.dtype)
            for component in _components(value)
        )
        return _from_components(value, components)


class _Slot:
    __slots__ = ("staging", "future")

    def __init__(self):
        self.staging = None
        self.future = None


class AsyncCacheWriter:
    """Copy compact K/V through a small pinned ring without manual residency."""

    def __init__(self, runtime, device, depth=2):
        self.runtime = runtime
        self.device = torch.device(device)
        self.depth = max(1, min(3, int(depth)))
        self.enabled = self.device.type == "cuda"
        self.disabled_reason = None
        self.stream = None
        self.executor = None
        self.slots = [_Slot() for _ in range(self.depth)]
        self.cursor = 0
        self.lock = threading.Lock()
        self.host_copy_seconds = 0.0
        self.enqueues = 0
        self.waits = 0
        self.bytes = 0

        if not self.enabled:
            self.disabled_reason = "cache source is not CUDA"
            return
        try:
            from comfy.cli_args import args
            if bool(getattr(args, "disable_pinned_memory", False)):
                self.enabled = False
                self.disabled_reason = "ComfyUI pinned memory is disabled"
                return
        except (ImportError, AttributeError):
            pass
        try:
            with torch.cuda.device(self.device):
                self.stream = torch.cuda.Stream(device=self.device)
                # Probe pinned allocation now so driver/configuration failures
                # select the synchronous path before cache state is pending.
                torch.empty((1,), dtype=torch.uint8, pin_memory=True)
            self.executor = ThreadPoolExecutor(
                max_workers=self.depth,
                thread_name_prefix="flashvsr-cache-write",
            )
        except Exception as error:
            self.enabled = False
            self.disabled_reason = f"pinned staging unavailable: {error}"
            self.stream = None
            self.executor = None

    @staticmethod
    def _matching(staging, sources):
        return (
            staging is not None
            and len(staging) == len(sources)
            and all(
                item.shape == source.shape and item.dtype == source.dtype
                for item, source in zip(staging, sources)
            )
        )

    @staticmethod
    def _copy_host(event, staging, destinations, sources):
        # Holding sources in this callable keeps their CUDA storage alive until
        # the transfer stream has consumed it.
        event.synchronize()
        del sources
        started = time.perf_counter()
        for source, destination in zip(staging, destinations):
            destination.copy_(source, non_blocking=False)
        return time.perf_counter() - started

    def _finish_slot(self, slot, wait_stage=True):
        if slot.future is None:
            return
        marker = (
            self.runtime.profile_start(self.device)
            if wait_stage else None
        )
        elapsed = slot.future.result()
        if marker is not None:
            self.runtime.profile_end("kv_write_wait_for_slot", marker)
        with self.lock:
            self.host_copy_seconds += float(elapsed)
        slot.future = None
        if wait_stage:
            self.waits += 1

    def _sync_pair(self, k_value, v_value):
        outputs = []
        for value in (k_value, v_value):
            destination = _empty_cpu_like(value)
            for source, target in zip(
                _components(value), _components(destination)
            ):
                target.copy_(source, non_blocking=False)
            outputs.append(destination)
        return tuple(outputs)

    def submit_pair(self, k_value, v_value):
        """Return ordinary CPU placeholders populated before :meth:`flush`."""
        if not self.enabled:
            return self._sync_pair(k_value, v_value)

        sources = (*_components(k_value), *_components(v_value))
        cpu_k = _empty_cpu_like(k_value)
        cpu_v = _empty_cpu_like(v_value)
        destinations = (*_components(cpu_k), *_components(cpu_v))
        slot = self.slots[self.cursor]
        self.cursor = (self.cursor + 1) % self.depth
        self._finish_slot(slot)

        if not self._matching(slot.staging, sources):
            try:
                slot.staging = tuple(
                    torch.empty(
                        source.shape,
                        device="cpu",
                        dtype=source.dtype,
                        pin_memory=True,
                    )
                    for source in sources
                )
            except Exception as error:
                self.enabled = False
                self.disabled_reason = f"pinned staging allocation failed: {error}"
                return self._sync_pair(k_value, v_value)

        with torch.cuda.device(self.device):
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self.stream.wait_event(ready)
            with torch.cuda.stream(self.stream):
                start.record(self.stream)
                for source, staging in zip(sources, slot.staging):
                    staging.copy_(source, non_blocking=True)
                end.record(self.stream)
            self.runtime.profile_record(
                "kv_write_d2h", start, end, self.device
            )

        slot.future = self.executor.submit(
            self._copy_host,
            end,
            slot.staging,
            destinations,
            sources,
        )
        self.enqueues += 1
        self.bytes += sum(
            source.numel() * source.element_size() for source in sources
        )
        self.runtime.profile_count("kv_write_enqueue", 1)
        self.runtime.profile_count("kv_write_bytes", self.bytes, replace=True)
        return cpu_k, cpu_v

    def flush(self):
        if not self.enabled:
            return
        for slot in self.slots:
            self._finish_slot(slot, wait_stage=False)

    def close(self):
        try:
            self.flush()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=False)
                self.executor = None

    def report(self):
        if self.enabled:
            print(
                "[FlashVSR] async cache write-through: "
                f"enqueues={self.enqueues}, waits={self.waits}, "
                f"transferred={self.bytes / (1024 ** 3):.2f} GiB, "
                f"host_copy={self.host_copy_seconds * 1000:.1f} ms."
            )
        elif self.disabled_reason:
            print(
                "[FlashVSR] async cache write-through inactive: "
                f"{self.disabled_reason}; using synchronous CPU copies."
            )
