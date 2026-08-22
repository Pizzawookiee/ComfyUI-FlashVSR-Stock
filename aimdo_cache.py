"""Optional AIMDO residency for mutable FlashVSR cache values.

The CPU value is always authoritative.  A dedicated, deprioritized VBAR is a
disposable GPU mirror: mappings can disappear at any time and are repopulated
when their AIMDO signature or FlashVSR content generation changes.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
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


def _copy_cpu_value(value, destination):
    if isinstance(value, Int8Carrier):
        value.copy_to(destination)
    else:
        destination.copy_(value, non_blocking=False)


class _Region:
    __slots__ = ("allocation", "shape", "dtype", "signature")

    def __init__(self, allocation, tensor):
        self.allocation = allocation
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.signature = None


class AimdoCachedValue:
    """CPU-owned cache value with an optional evictable GPU mirror."""

    __slots__ = (
        "controller", "cpu_value", "regions", "generation",
        "resident_generation",
    )

    def __init__(self, controller, cpu_value, gpu_source=None):
        self.controller = controller
        self.cpu_value = cpu_value
        self.regions = [
            controller.allocate_region(component)
            for component in _components(cpu_value)
        ]
        self.generation = 1
        self.resident_generation = 0
        if gpu_source is not None:
            self._populate_if_available(gpu_source, self.generation, write=True)

    def _fault_all(self):
        controller = self.controller
        if not controller.enabled:
            return None, False
        started = time.perf_counter()
        faulted = []
        mapping_current = True
        try:
            for region in self.regions:
                signature = controller.vbar_fault(region.allocation)
                if signature is None:
                    controller.fault_misses += 1
                    controller.note_fault(False)
                    for existing, _tensor in faulted:
                        controller.vbar_unpin(existing.allocation)
                    return None, False
                mapping_current = (
                    mapping_current
                    and controller.signature_compare(
                        signature, region.signature
                    )
                )
                try:
                    tensor = controller.tensor_for(region)
                except Exception:
                    controller.vbar_unpin(region.allocation)
                    raise
                faulted.append((region, tensor))
                controller.note_fault(True)
                region.signature = signature
        except Exception as error:
            for region, _tensor in faulted:
                try:
                    controller.vbar_unpin(region.allocation)
                except Exception:
                    pass
            controller.disable(f"AIMDO cache fault failed: {error}")
            return None, False
        finally:
            controller.fault_seconds += time.perf_counter() - started
        return faulted, mapping_current

    def _unpin_all(self, faulted):
        if faulted is None:
            return
        for region, _tensor in faulted:
            try:
                self.controller.vbar_unpin(region.allocation)
            except Exception as error:
                self.controller.disable(
                    f"AIMDO cache unpin failed: {error}"
                )

    def _populate_if_available(self, source, generation, write=False):
        faulted, _mapping_current = self._fault_all()
        if faulted is None:
            return False
        try:
            source_components = _components(source)
            for (_region, destination), component in zip(
                faulted, source_components
            ):
                destination.copy_(
                    component.to(
                        device=destination.device,
                        dtype=destination.dtype,
                    ),
                    non_blocking=False,
                )
                self.controller.populated_bytes += (
                    destination.numel() * destination.element_size()
                )
            self.resident_generation = generation
            if write:
                self.controller.write_throughs += 1
            else:
                self.controller.rehydrates += 1
            return True
        except Exception as error:
            self.controller.disable(
                f"AIMDO cache population failed: {error}"
            )
            return False
        finally:
            self._unpin_all(faulted)

    def copy_to(self, destination):
        controller = self.controller
        controller.accesses += 1
        faulted, mapping_current = self._fault_all()
        if faulted is None:
            controller.cpu_fallbacks += 1
            _copy_cpu_value(self.cpu_value, destination)
            return
        try:
            if (
                not mapping_current
                or self.resident_generation != self.generation
            ):
                for (_region, gpu_tensor), cpu_tensor in zip(
                    faulted, _components(self.cpu_value)
                ):
                    gpu_tensor.copy_(cpu_tensor, non_blocking=False)
                    controller.populated_bytes += (
                        gpu_tensor.numel() * gpu_tensor.element_size()
                    )
                self.resident_generation = self.generation
                controller.rehydrates += 1
            else:
                controller.resident_hits += 1
            gpu_value = _from_components(
                self.cpu_value,
                tuple(tensor for _region, tensor in faulted),
            )
            _copy_cpu_value(gpu_value, destination)
        except Exception as error:
            controller.disable(f"AIMDO cache staging failed: {error}")
            controller.cpu_fallbacks += 1
            _copy_cpu_value(self.cpu_value, destination)
        finally:
            self._unpin_all(faulted)

    @contextmanager
    def acquire_compact(self, device):
        """Yield the compact value on *device* while AIMDO pages stay pinned.

        Unlike :meth:`copy_to`, this does not expand an Int8Carrier into a
        floating-point destination.  The caller can therefore dequantize
        directly into its final kernel layout.  CPU remains authoritative and
        is used transparently when the evictable AIMDO mapping is absent.
        """
        controller = self.controller
        controller.accesses += 1
        target = torch.device(device)
        faulted, mapping_current = self._fault_all()
        if faulted is None:
            controller.cpu_fallbacks += 1
            components = tuple(
                component.to(device=target, non_blocking=False)
                for component in _components(self.cpu_value)
            )
            try:
                yield _from_components(self.cpu_value, components)
            finally:
                del components
            return

        gpu_value = None
        try:
            if (
                not mapping_current
                or self.resident_generation != self.generation
            ):
                for (_region, gpu_tensor), cpu_tensor in zip(
                    faulted, _components(self.cpu_value)
                ):
                    gpu_tensor.copy_(cpu_tensor, non_blocking=False)
                    controller.populated_bytes += (
                        gpu_tensor.numel() * gpu_tensor.element_size()
                    )
                self.resident_generation = self.generation
                controller.rehydrates += 1
            else:
                controller.resident_hits += 1
            gpu_value = _from_components(
                self.cpu_value,
                tuple(tensor for _region, tensor in faulted),
            )
        except Exception as error:
            controller.disable(
                f"AIMDO compact cache access failed: {error}"
            )
            controller.cpu_fallbacks += 1
            self._unpin_all(faulted)
            faulted = None
            components = tuple(
                component.to(device=target, non_blocking=False)
                for component in _components(self.cpu_value)
            )
            try:
                yield _from_components(self.cpu_value, components)
            finally:
                del components
            return

        try:
            yield gpu_value
        finally:
            self._unpin_all(faulted)

    def prepare_update(self, cpu_value, gpu_source=None):
        generation = self.generation + 1
        if gpu_source is not None:
            self._populate_if_available(
                gpu_source, generation, write=True
            )
        return cpu_value, generation

    def commit_update(self, pending):
        self.cpu_value, self.generation = pending


class AimdoCacheController:
    """Own a lower-priority cache VBAR and its diagnostic counters."""

    def __init__(self, total_bytes, allocation_count, device, runtime=None):
        self.device = torch.device(device)
        self.runtime = runtime
        self.enabled = False
        self.disabled_reason = None
        self.vbar = None
        self.vbar_fault = None
        self.vbar_unpin = None
        self.signature_compare = None
        self.aimdo_to_tensor = None
        self.accesses = 0
        self.resident_hits = 0
        self.fault_misses = 0
        self.cpu_fallbacks = 0
        self.rehydrates = 0
        self.write_throughs = 0
        self.populated_bytes = 0
        self.fault_seconds = 0.0

        try:
            import comfy.memory_management
            if not bool(comfy.memory_management.aimdo_enabled):
                self.disabled_reason = "ComfyUI AIMDO is not enabled"
                return
            if self.device.type != "cuda":
                self.disabled_reason = "AIMDO cache residency requires CUDA"
                return
            import comfy_aimdo.model_vbar as model_vbar
            import comfy_aimdo.torch as aimdo_torch

            # One packed VBAR avoids rounding each K/V carrier to AIMDO's
            # 32-MiB page granularity.  Allow 512-byte allocation alignment
            # and one final guard page without making the physical pages
            # resident up front.
            alignment_slack = max(4096, int(allocation_count) * 512)
            page = 32 * 1024 * 1024
            capacity = math.ceil(
                (int(total_bytes) + alignment_slack + page) / page
            ) * page
            self.vbar = model_vbar.ModelVBAR(
                capacity, self.device.index
            )
            # A cache miss always has a correct CPU route. Keep model weights
            # above this opportunistic mirror in AIMDO's priority order.
            self.vbar.deprioritize()
            self.vbar_fault = model_vbar.vbar_fault
            self.vbar_unpin = model_vbar.vbar_unpin
            self.signature_compare = model_vbar.vbar_signature_compare
            self.aimdo_to_tensor = aimdo_torch.aimdo_to_tensor
            self.enabled = True
        except Exception as error:
            self.disabled_reason = f"AIMDO cache initialization failed: {error}"
            self.vbar = None

    def note_fault(self, success):
        runtime = self.runtime
        if runtime is None:
            return
        runtime.profile_count(
            "aimdo_fault_success" if success else "aimdo_fault_failure", 1
        )

    def allocate_region(self, tensor):
        if not self.enabled:
            raise RuntimeError(self.disabled_reason or "AIMDO cache disabled")
        size = tensor.numel() * tensor.element_size()
        allocation = self.vbar.alloc(size)
        return _Region(allocation, tensor)

    def tensor_for(self, region):
        raw = self.aimdo_to_tensor(region.allocation, self.device)
        return raw.view(dtype=region.dtype).view(region.shape)

    def wrap(self, cpu_value, gpu_source=None):
        if not self.enabled:
            return cpu_value
        try:
            return AimdoCachedValue(self, cpu_value, gpu_source)
        except Exception as error:
            self.disable(f"AIMDO cache allocation failed: {error}")
            return cpu_value

    def disable(self, reason):
        if self.disabled_reason is None:
            self.disabled_reason = str(reason)
            print(f"[FlashVSR] {self.disabled_reason}; using CPU cache.")
        self.enabled = False

    def report(self):
        if self.vbar is None:
            if self.disabled_reason:
                print(
                    f"[FlashVSR] AIMDO cache inactive: "
                    f"{self.disabled_reason}."
                )
            return
        hit_rate = (
            100.0 * self.resident_hits / self.accesses
            if self.accesses else 0.0
        )
        try:
            resident_bytes = int(self.vbar.loaded_size())
        except Exception:
            resident_bytes = 0
        print(
            "[FlashVSR] AIMDO cache: "
            f"accesses={self.accesses}, resident_hits={self.resident_hits} "
            f"({hit_rate:.1f}%), rehydrates={self.rehydrates}, "
            f"fault_misses={self.fault_misses}, "
            f"CPU fallbacks={self.cpu_fallbacks}, "
            f"write_throughs={self.write_throughs}, "
            f"resident={resident_bytes / (1024 ** 2):.0f} MiB, "
            f"populated={self.populated_bytes / (1024 ** 3):.2f} GiB, "
            f"fault_host_time={self.fault_seconds * 1000:.1f} ms."
        )


def is_aimdo_value(value):
    return isinstance(value, AimdoCachedValue)
