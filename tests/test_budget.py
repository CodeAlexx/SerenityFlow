from __future__ import annotations
import pytest
from serenityflow.core.budget import MemoryBudget, ResidencyClass, Allocation

GB = 1024 * 1024 * 1024
MB = 1024 * 1024


class TestMemoryBudget:
    def test_initial_available(self):
        b = MemoryBudget(24 * GB, reserved_bytes=2 * GB)
        assert b.available == 22 * GB

    def test_allocate_within_budget(self):
        b = MemoryBudget(24 * GB)
        ok = b.allocate("a1", "flux", 10 * GB, ResidencyClass.FULL_RESIDENT)
        assert ok
        assert b.available == 14 * GB

    def test_allocate_over_budget(self):
        b = MemoryBudget(24 * GB)
        ok = b.allocate("a1", "flux", 25 * GB, ResidencyClass.FULL_RESIDENT)
        assert not ok
        assert len(b.allocations) == 0

    def test_release_frees_space(self):
        b = MemoryBudget(24 * GB)
        b.allocate("a1", "flux", 10 * GB, ResidencyClass.FULL_RESIDENT)
        released = b.release("a1")
        assert released == 10 * GB
        assert b.available == 24 * GB

    def test_pinned_cpu_doesnt_count(self):
        b = MemoryBudget(24 * GB)
        b.allocate("cpu1", "vae", 2 * GB, ResidencyClass.PINNED_CPU)
        assert b.available == 24 * GB
        assert b.current_allocated == 0

    def test_reserved_not_evictable(self):
        b = MemoryBudget(24 * GB)
        b.allocate("r1", "taesd", 100 * MB, ResidencyClass.RESERVED)
        b.set_active("r1", False)
        candidates = b.eviction_candidates()
        assert len(candidates) == 0

    def test_eviction_priority_order(self):
        b = MemoryBudget(24 * GB)
        b.allocate("stream1", "flux", 2 * GB, ResidencyClass.STREAM_BLOCKS)
        b.allocate("hook1", "cn", 1 * GB, ResidencyClass.INFERENCE_HOOK)
        b.allocate("full1", "vae", 500 * MB, ResidencyClass.FULL_RESIDENT)
        b.set_active("stream1", False)
        b.set_active("hook1", False)
        b.set_active("full1", False)
        candidates = b.eviction_candidates()
        classes = [c.residency for c in candidates]
        assert classes == [
            ResidencyClass.STREAM_BLOCKS,
            ResidencyClass.INFERENCE_HOOK,
            ResidencyClass.FULL_RESIDENT,
        ]

    def test_active_excluded_from_eviction(self):
        b = MemoryBudget(24 * GB)
        b.allocate("a1", "flux", 10 * GB, ResidencyClass.FULL_RESIDENT)
        # active=True by default
        assert len(b.eviction_candidates()) == 0

    def test_evict_to_fit(self):
        b = MemoryBudget(24 * GB)
        b.allocate("a1", "m1", 10 * GB, ResidencyClass.STREAM_BLOCKS)
        b.allocate("a2", "m2", 10 * GB, ResidencyClass.STREAM_BLOCKS)
        b.set_active("a1", False)
        b.set_active("a2", False)
        evicted = b.evict_to_fit(15 * GB)
        assert len(evicted) >= 1
        assert b.available >= 15 * GB

    def test_evict_to_fit_returns_ids(self):
        b = MemoryBudget(10 * GB)
        b.allocate("a1", "m1", 5 * GB, ResidencyClass.STREAM_BLOCKS)
        b.set_active("a1", False)
        evicted = b.evict_to_fit(8 * GB)
        assert "a1" in evicted

    def test_stats(self):
        b = MemoryBudget(24 * GB, reserved_bytes=2 * GB)
        b.allocate("a1", "flux", 10 * GB, ResidencyClass.FULL_RESIDENT)
        s = b.stats()
        assert s["total_mb"] == round(24 * GB / MB, 1)
        assert s["reserved_mb"] == round(2 * GB / MB, 1)
        assert s["allocated_mb"] == round(10 * GB / MB, 1)
        assert s["allocations"] == 1

    def test_release_nonexistent(self):
        b = MemoryBudget(24 * GB)
        assert b.release("nonexistent") == 0
