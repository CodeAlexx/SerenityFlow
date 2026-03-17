"""Output cache with IS_CHANGED support and residency-aware tensor management.

Signature = hash(class_type, literal inputs, upstream signatures, IS_CHANGED result).
Cache invalidation is per-node. LRU eviction when over capacity.

When a MemoryBudget is provided, cached GPU tensors are tracked and can be
offloaded to CPU-pinned memory under VRAM pressure.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

import torch

from serenityflow.core.budget import MemoryBudget, ResidencyClass


@dataclass
class CachedOutput:
    """Cached result from a node execution."""
    outputs: Any
    ui: dict
    signature: str


class CacheStore:
    """LRU cache for node outputs keyed by node_id.

    Optionally tracks GPU tensor residency via MemoryBudget.
    """

    def __init__(self, max_entries: int = 256, budget: MemoryBudget | None = None):
        self.cache: dict[str, CachedOutput] = {}
        self.max_entries = max_entries
        self._access_order: list[str] = []
        self.budget = budget
        self._tensor_locations: dict[str, str] = {}  # node_id -> "gpu" | "cpu_pinned" | "cpu"

    def get(self, node_id: str) -> Optional[CachedOutput]:
        """Get cached output for node. Returns None if not cached."""
        cached = self.cache.get(node_id)
        if cached is not None:
            # Move to end (most recently used)
            if node_id in self._access_order:
                self._access_order.remove(node_id)
            self._access_order.append(node_id)
        return cached

    def set(self, node_id: str, outputs: Any, ui: dict, signature: str) -> None:
        """Store node output with its cache signature."""
        # Release old budget allocation if overwriting
        if node_id in self._tensor_locations and self.budget is not None:
            if self._tensor_locations[node_id] == "gpu":
                self.budget.release(f"cache:{node_id}")
            self._tensor_locations.pop(node_id, None)

        self.cache[node_id] = CachedOutput(outputs, ui, signature)
        if node_id in self._access_order:
            self._access_order.remove(node_id)
        self._access_order.append(node_id)

        # Track tensor residency
        if self.budget is not None and self._has_tensors(outputs):
            tensor_bytes = self._tensor_bytes(outputs)
            if tensor_bytes > 0 and self.budget.can_fit(tensor_bytes):
                self.budget.allocate(
                    f"cache:{node_id}", f"cache:{node_id}",
                    tensor_bytes, ResidencyClass.FULL_RESIDENT,
                )
                self._tensor_locations[node_id] = "gpu"
            else:
                self._tensor_locations[node_id] = "cpu"

        self._evict()

    def invalidate(self, node_id: str) -> None:
        """Remove a specific node from cache."""
        # Release budget
        if self.budget is not None and self._tensor_locations.get(node_id) == "gpu":
            self.budget.release(f"cache:{node_id}")
        self._tensor_locations.pop(node_id, None)

        self.cache.pop(node_id, None)
        if node_id in self._access_order:
            self._access_order.remove(node_id)

    def clear(self) -> None:
        """Clear all cached outputs."""
        # Release all budget allocations
        if self.budget is not None:
            for node_id, loc in self._tensor_locations.items():
                if loc == "gpu":
                    self.budget.release(f"cache:{node_id}")
        self._tensor_locations.clear()

        self.cache.clear()
        self._access_order.clear()

    def all_node_ids(self) -> set[str]:
        """Return set of all cached node IDs."""
        return set(self.cache.keys())

    # ─── Residency management ───

    def offload_to_cpu(self, node_id: str) -> int:
        """Move cached tensors from GPU to CPU-pinned. Frees GPU budget.

        Returns bytes freed.
        """
        cached = self.cache.get(node_id)
        if cached is None or self._tensor_locations.get(node_id) != "gpu":
            return 0

        cached.outputs, freed = self._move_tensors_to_cpu(cached.outputs)
        if self.budget is not None:
            self.budget.release(f"cache:{node_id}")
        self._tensor_locations[node_id] = "cpu_pinned"
        return freed

    def ensure_gpu(self, node_id: str) -> None:
        """Move cached tensors back to GPU if they were offloaded."""
        cached = self.cache.get(node_id)
        if cached is None:
            return
        loc = self._tensor_locations.get(node_id, "cpu")
        if loc == "gpu":
            return
        cached.outputs = self._move_tensors_to_gpu(cached.outputs)
        tensor_bytes = self._tensor_bytes(cached.outputs)
        if self.budget is not None and tensor_bytes > 0:
            self.budget.allocate(
                f"cache:{node_id}", f"cache:{node_id}",
                tensor_bytes, ResidencyClass.FULL_RESIDENT,
            )
        self._tensor_locations[node_id] = "gpu"

    def pressure_evict(self, needed_bytes: int) -> int:
        """Offload cached GPU tensors to CPU to free VRAM. Returns bytes freed."""
        freed = 0
        for node_id in list(self._access_order):
            if freed >= needed_bytes:
                break
            if self._tensor_locations.get(node_id) == "gpu":
                freed += self.offload_to_cpu(node_id)
        return freed

    def get_location(self, node_id: str) -> str | None:
        """Return tensor location for a cached node, or None if not tracked."""
        return self._tensor_locations.get(node_id)

    # ─── Private ───

    def _evict(self) -> None:
        """Evict oldest entries when over capacity."""
        while len(self.cache) > self.max_entries and self._access_order:
            oldest = self._access_order.pop(0)
            # Release budget for evicted node
            if self.budget is not None and self._tensor_locations.get(oldest) == "gpu":
                self.budget.release(f"cache:{oldest}")
            self._tensor_locations.pop(oldest, None)
            self.cache.pop(oldest, None)

    @staticmethod
    def _has_tensors(outputs) -> bool:
        """Check if outputs contain any tensors."""
        found = False

        def visit(value):
            nonlocal found
            if found:
                return
            if isinstance(value, torch.Tensor):
                found = True
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(outputs)
        return found

    @staticmethod
    def _tensor_bytes(outputs) -> int:
        """Sum bytes of all tensors in outputs."""
        total = 0

        def visit(value):
            nonlocal total
            if isinstance(value, torch.Tensor):
                total += value.nelement() * value.element_size()
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(outputs)
        return total

    @staticmethod
    def _move_tensors_to_cpu(outputs) -> tuple[Any, int]:
        """Walk output structure, move GPU tensors to CPU pinned. Returns bytes freed."""
        moved, freed = CacheStore._transform_tensors(outputs, to_cpu=True)
        return moved, freed

    @staticmethod
    def _move_tensors_to_gpu(outputs) -> Any:
        """Walk output structure, move CPU tensors to GPU."""
        moved, _ = CacheStore._transform_tensors(outputs, to_cpu=False)
        return moved

    @staticmethod
    def _transform_tensors(outputs, *, to_cpu: bool):
        """Recursively transform tensors while preserving container types."""
        freed = 0
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def transform(value):
            nonlocal freed
            if isinstance(value, torch.Tensor):
                if to_cpu:
                    if not value.is_cuda:
                        return value
                    nbytes = value.nelement() * value.element_size()
                    cpu_tensor = torch.empty(
                        value.shape,
                        dtype=value.dtype,
                        pin_memory=True,
                    )
                    cpu_tensor.copy_(value)
                    freed += nbytes
                    return cpu_tensor

                if value.is_cuda:
                    return value
                return value.to(device, non_blocking=True)

            if isinstance(value, dict):
                return {k: transform(v) for k, v in value.items()}

            if isinstance(value, list):
                return [transform(item) for item in value]

            if isinstance(value, tuple):
                return tuple(transform(item) for item in value)

            return value

        return transform(outputs), freed


def compute_signature(
    class_type: str,
    inputs: dict,
    upstream_sigs: dict,
    is_changed_result: Any = None,
) -> str:
    """Compute deterministic cache signature for a node.

    Hashes class_type + literal scalar inputs + upstream node signatures
    + IS_CHANGED result. Link inputs are captured via upstream_sigs.
    """
    h = hashlib.sha256()
    h.update(class_type.encode())

    # Literal/scalar inputs (sorted for determinism)
    for key in sorted(inputs.keys()):
        val = inputs[key]
        if isinstance(val, (str, int, float, bool)):
            h.update(f"{key}={val}".encode())

    # Upstream signatures (sorted)
    for key in sorted(upstream_sigs.keys()):
        h.update(f"upstream:{key}={upstream_sigs[key]}".encode())

    # IS_CHANGED
    if is_changed_result is not None:
        h.update(f"is_changed={is_changed_result}".encode())

    return h.hexdigest()[:16]


__all__ = ["CachedOutput", "CacheStore", "compute_signature"]
