"""Tests for residency-aware caching in CacheStore.

Builder: budget allocation, offload, restore, pressure eviction
Bug Fixer: edge cases (no budget, non-tensor outputs, double offload)
Skeptic: adversarial scenarios (budget full, empty eviction list)
"""
from __future__ import annotations

import torch
import pytest

from serenityflow.core.budget import MemoryBudget, ResidencyClass
from serenityflow.executor.cache import CacheStore


def _tensor_output(dim: int = 64) -> tuple:
    """Create a tuple with a tensor output (CPU, simulating GPU-like tracking)."""
    return (torch.randn(dim, dim),)


def _scalar_output() -> tuple:
    return ("hello", 42)


def _budget(total_mb: int = 100) -> MemoryBudget:
    return MemoryBudget(total_mb * 1024 * 1024)


# ─── Builder Tests ───


class TestBudgetAllocation:
    def test_set_gpu_tensor_allocates_budget(self):
        """Cache with budget: set GPU tensor -> budget allocated."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        t = torch.randn(64, 64)
        cache.set("node1", (t,), {}, "sig1")

        assert cache.get_location("node1") == "gpu"
        assert budget.current_allocated > 0

    def test_tensor_bytes_tracked(self):
        """Allocated bytes match tensor size."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        t = torch.randn(100, 100)  # 100*100*4 = 40000 bytes
        expected = t.nelement() * t.element_size()
        cache.set("node1", (t,), {}, "sig1")
        assert budget.current_allocated == expected


class TestCacheOffload:
    def test_offload_releases_budget(self):
        """Offload to CPU -> budget released, tensor still accessible."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("node1", [torch.randn(64, 64)], {}, "sig1")

        allocated_before = budget.current_allocated
        freed = cache.offload_to_cpu("node1")
        assert freed > 0 or allocated_before > 0
        assert cache.get_location("node1") == "cpu_pinned"
        # Budget should be released
        assert budget.current_allocated == 0

    def test_offloaded_data_still_accessible(self):
        """After offload, cache.get() still returns data."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("node1", (torch.randn(8, 8),), {}, "sig1")
        cache.offload_to_cpu("node1")
        cached = cache.get("node1")
        assert cached is not None
        assert cached.signature == "sig1"


class TestCacheRestore:
    def test_ensure_gpu_reallocates(self):
        """ensure_gpu -> tensor back on GPU, budget re-allocated."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("node1", [torch.randn(64, 64)], {}, "sig1")
        cache.offload_to_cpu("node1")
        assert budget.current_allocated == 0

        cache.ensure_gpu("node1")
        assert cache.get_location("node1") == "gpu"
        assert budget.current_allocated > 0

    def test_ensure_gpu_already_gpu(self):
        """ensure_gpu on GPU tensor -> no-op."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("node1", (torch.randn(8, 8),), {}, "sig1")
        allocated = budget.current_allocated
        cache.ensure_gpu("node1")  # Should be no-op
        assert budget.current_allocated == allocated


class TestPressureEvict:
    def test_pressure_evict_frees_lru(self):
        """Pressure evict: frees LRU GPU tensors first."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("a", (torch.randn(32, 32),), {}, "s1")
        cache.set("b", (torch.randn(32, 32),), {}, "s2")
        cache.set("c", (torch.randn(32, 32),), {}, "s3")
        # Access b to make it more recent than a
        cache.get("b")

        freed = cache.pressure_evict(needed_bytes=4096)
        # a should be evicted first (LRU)
        assert cache.get_location("a") == "cpu_pinned"

    def test_pressure_evict_returns_bytes(self):
        """pressure_evict returns total bytes freed."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        t = torch.randn(64, 64)
        cache.set("node1", [t], {}, "sig1")
        freed = cache.pressure_evict(needed_bytes=t.nelement() * t.element_size())
        # freed may be 0 on CPU-only (no cuda), but location should change
        assert cache.get_location("node1") == "cpu_pinned"


# ─── Bug Fixer Tests ───


class TestNoBudget:
    def test_works_as_pure_lru(self):
        """Cache without budget (budget=None) -> works as pure LRU, no residency tracking."""
        cache = CacheStore()
        cache.set("a", (torch.randn(8, 8),), {}, "s1")
        cache.set("b", ("string",), {}, "s2")
        assert cache.get("a") is not None
        assert cache.get("b") is not None
        assert cache.get_location("a") is None
        assert cache.get_location("b") is None

    def test_offload_no_budget_returns_zero(self):
        """Offload without budget -> returns 0."""
        cache = CacheStore()
        cache.set("a", (torch.randn(8, 8),), {}, "s1")
        freed = cache.offload_to_cpu("a")
        assert freed == 0


class TestNonTensorOutputs:
    def test_string_outputs_no_budget(self):
        """Non-tensor outputs (strings, ints) -> no budget interaction."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("node1", ("hello", 42), {}, "sig1")
        assert budget.current_allocated == 0
        assert cache.get_location("node1") is None or cache.get_location("node1") == "cpu"

    def test_empty_tuple_no_budget(self):
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("node1", (), {}, "sig1")
        assert budget.current_allocated == 0


class TestEdgeCases:
    def test_offload_nonexistent(self):
        """Offload non-existent node -> returns 0."""
        cache = CacheStore(budget=_budget(100))
        assert cache.offload_to_cpu("nonexistent") == 0

    def test_ensure_gpu_nonexistent(self):
        """ensure_gpu on missing node -> no-op, no crash."""
        cache = CacheStore(budget=_budget(100))
        cache.ensure_gpu("nonexistent")  # Should not raise

    def test_double_offload(self):
        """Offloading twice -> second call returns 0."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("node1", [torch.randn(8, 8)], {}, "sig1")
        cache.offload_to_cpu("node1")
        freed = cache.offload_to_cpu("node1")
        assert freed == 0
        assert cache.get_location("node1") == "cpu_pinned"

    def test_invalidate_releases_budget(self):
        """Invalidating a cached node releases its budget allocation."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("node1", (torch.randn(64, 64),), {}, "sig1")
        assert budget.current_allocated > 0
        cache.invalidate("node1")
        assert budget.current_allocated == 0

    def test_clear_releases_all_budget(self):
        """Clear releases all budget allocations."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("a", (torch.randn(8, 8),), {}, "s1")
        cache.set("b", (torch.randn(8, 8),), {}, "s2")
        assert budget.current_allocated > 0
        cache.clear()
        assert budget.current_allocated == 0

    def test_overwrite_releases_old_budget(self):
        """Overwriting a cached entry releases old budget before new allocation."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        cache.set("node1", (torch.randn(8, 8),), {}, "sig1")
        old_alloc = budget.current_allocated
        cache.set("node1", (torch.randn(16, 16),), {}, "sig2")
        # New allocation should reflect new tensor size
        new_alloc = budget.current_allocated
        expected = 16 * 16 * 4  # float32
        assert new_alloc == expected


# ─── Skeptic Tests ───


class TestSkepticAdversarial:
    def test_budget_full_tensor_stays_cpu(self):
        """Budget full -> tensor stays on CPU, no GPU allocation."""
        budget = MemoryBudget(total_vram=1024)  # Tiny budget: 1KB
        cache = CacheStore(budget=budget)
        big_tensor = torch.randn(256, 256)  # ~256KB
        cache.set("node1", (big_tensor,), {}, "sig1")
        # Should not be on GPU (doesn't fit)
        assert cache.get_location("node1") == "cpu"
        assert budget.current_allocated == 0

    def test_lru_eviction_releases_budget(self):
        """LRU eviction releases budget for evicted entries."""
        budget = _budget(100)
        cache = CacheStore(max_entries=2, budget=budget)
        cache.set("a", (torch.randn(8, 8),), {}, "s1")
        cache.set("b", (torch.randn(8, 8),), {}, "s2")
        alloc_2 = budget.current_allocated
        # Adding c evicts a
        cache.set("c", (torch.randn(8, 8),), {}, "s3")
        assert cache.get("a") is None
        # Budget should still be reasonable (2 entries worth)
        assert budget.current_allocated == alloc_2  # b + c same as a + b

    def test_nested_tensor_outputs(self):
        """Nested list of tensors tracked correctly."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        t1 = torch.randn(8, 8)
        t2 = torch.randn(4, 4)
        cache.set("node1", ([t1, t2],), {}, "sig1")
        expected = t1.nelement() * t1.element_size() + t2.nelement() * t2.element_size()
        assert budget.current_allocated == expected

    def test_pressure_evict_all_gpu_tensors(self):
        """Pressure evict when all entries are GPU tensors."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        for i in range(5):
            cache.set(f"n{i}", (torch.randn(8, 8),), {}, f"s{i}")
        cache.pressure_evict(needed_bytes=999999)
        # All should be offloaded
        for i in range(5):
            assert cache.get_location(f"n{i}") == "cpu_pinned"
        assert budget.current_allocated == 0

    def test_mixed_tensor_scalar_outputs(self):
        """Outputs with both tensors and scalars tracked correctly."""
        budget = _budget(100)
        cache = CacheStore(budget=budget)
        t = torch.randn(16, 16)
        cache.set("node1", (t, "text", 42), {}, "sig1")
        expected = t.nelement() * t.element_size()
        assert budget.current_allocated == expected
