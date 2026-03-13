from __future__ import annotations

"""
All GPU memory goes through here. No subsystem allocates independently.
Stagehand is the enforcement layer — this is the policy layer.
"""
from dataclasses import dataclass
from enum import Enum

class ResidencyClass(Enum):
    STREAM_BLOCKS = "stream_blocks"     # Block-swap model, low eviction priority
    FULL_RESIDENT = "full_resident"     # Entire model on GPU
    RESERVED = "reserved"               # Permanent (TAESD, constants), never evict
    INFERENCE_HOOK = "inference_hook"   # ControlNet/IP-Adapter, evict when inactive
    PINNED_CPU = "pinned_cpu"           # CPU-pinned, not on GPU budget

@dataclass
class Allocation:
    alloc_id: str
    model_id: str
    size_bytes: int
    residency: ResidencyClass
    active: bool = True  # False = eviction candidate

class MemoryBudget:
    def __init__(self, total_vram: int, reserved_bytes: int = 0):
        self.total_vram = total_vram
        self.reserved = reserved_bytes  # For activations, TAESD, overhead
        self.model_budget = total_vram - reserved_bytes
        self.allocations: dict[str, Allocation] = {}

    @property
    def current_allocated(self) -> int:
        return sum(a.size_bytes for a in self.allocations.values()
                   if a.residency != ResidencyClass.PINNED_CPU)

    @property
    def available(self) -> int:
        return max(0, self.model_budget - self.current_allocated)

    def can_fit(self, size_bytes: int) -> bool:
        return self.available >= size_bytes

    def allocate(self, alloc_id: str, model_id: str, size_bytes: int,
                 residency: ResidencyClass) -> bool:
        if residency == ResidencyClass.PINNED_CPU:
            self.allocations[alloc_id] = Allocation(alloc_id, model_id, size_bytes, residency)
            return True
        if not self.can_fit(size_bytes):
            return False
        self.allocations[alloc_id] = Allocation(alloc_id, model_id, size_bytes, residency)
        return True

    def release(self, alloc_id: str) -> int:
        alloc = self.allocations.pop(alloc_id, None)
        return alloc.size_bytes if alloc else 0

    def set_active(self, alloc_id: str, active: bool) -> None:
        if alloc_id in self.allocations:
            self.allocations[alloc_id].active = active

    def eviction_candidates(self) -> list[Allocation]:
        """Return evictable allocations, lowest priority first."""
        priority = {
            ResidencyClass.STREAM_BLOCKS: 0,
            ResidencyClass.INFERENCE_HOOK: 1,
            ResidencyClass.FULL_RESIDENT: 2,
            ResidencyClass.RESERVED: 999,
            ResidencyClass.PINNED_CPU: 999,
        }
        candidates = [a for a in self.allocations.values()
                      if not a.active
                      and a.residency not in (ResidencyClass.RESERVED, ResidencyClass.PINNED_CPU)]
        candidates.sort(key=lambda a: (priority.get(a.residency, 99), -a.size_bytes))
        return candidates

    def evict_to_fit(self, needed_bytes: int) -> list[str]:
        """Evict until `needed_bytes` available. Returns list of evicted alloc_ids."""
        evicted = []
        for candidate in self.eviction_candidates():
            if self.available >= needed_bytes:
                break
            evicted.append(candidate.alloc_id)
            self.release(candidate.alloc_id)
        return evicted

    def stats(self) -> dict:
        return {
            "total_mb": round(self.total_vram / 1024 / 1024, 1),
            "reserved_mb": round(self.reserved / 1024 / 1024, 1),
            "budget_mb": round(self.model_budget / 1024 / 1024, 1),
            "allocated_mb": round(self.current_allocated / 1024 / 1024, 1),
            "available_mb": round(self.available / 1024 / 1024, 1),
            "allocations": len(self.allocations),
        }
