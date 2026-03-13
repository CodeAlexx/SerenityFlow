"""Tests for executor output cache."""
from __future__ import annotations

import pytest

from serenityflow.executor.cache import CacheStore, CachedOutput, compute_signature


class TestCacheStore:
    def test_set_and_get(self):
        cache = CacheStore()
        cache.set("node1", ("output",), {}, "sig1")
        cached = cache.get("node1")
        assert cached is not None
        assert cached.outputs == ("output",)
        assert cached.signature == "sig1"

    def test_get_missing_returns_none(self):
        cache = CacheStore()
        assert cache.get("nonexistent") is None

    def test_invalidate_removes(self):
        cache = CacheStore()
        cache.set("node1", ("out",), {}, "sig")
        cache.invalidate("node1")
        assert cache.get("node1") is None

    def test_invalidate_missing_no_error(self):
        cache = CacheStore()
        cache.invalidate("nonexistent")  # Should not raise

    def test_clear_removes_all(self):
        cache = CacheStore()
        cache.set("a", (), {}, "s1")
        cache.set("b", (), {}, "s2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert len(cache.all_node_ids()) == 0

    def test_all_node_ids(self):
        cache = CacheStore()
        cache.set("a", (), {}, "s1")
        cache.set("b", (), {}, "s2")
        assert cache.all_node_ids() == {"a", "b"}

    def test_lru_eviction(self):
        cache = CacheStore(max_entries=3)
        cache.set("a", (), {}, "s1")
        cache.set("b", (), {}, "s2")
        cache.set("c", (), {}, "s3")
        # Access a to make it recent
        cache.get("a")
        # Add d — should evict b (oldest after a was accessed)
        cache.set("d", (), {}, "s4")
        assert cache.get("b") is None  # evicted
        assert cache.get("a") is not None  # still present (was accessed)
        assert cache.get("c") is not None
        assert cache.get("d") is not None

    def test_overwrite_existing(self):
        cache = CacheStore()
        cache.set("node1", ("old",), {}, "sig1")
        cache.set("node1", ("new",), {}, "sig2")
        cached = cache.get("node1")
        assert cached.outputs == ("new",)
        assert cached.signature == "sig2"

    def test_ui_stored(self):
        cache = CacheStore()
        ui = {"images": ["test.png"]}
        cache.set("node1", (), ui, "sig")
        assert cache.get("node1").ui == ui


class TestComputeSignature:
    def test_same_inputs_same_sig(self):
        s1 = compute_signature("KSampler", {"seed": 42, "steps": 20}, {})
        s2 = compute_signature("KSampler", {"seed": 42, "steps": 20}, {})
        assert s1 == s2

    def test_different_inputs_different_sig(self):
        s1 = compute_signature("KSampler", {"seed": 42}, {})
        s2 = compute_signature("KSampler", {"seed": 43}, {})
        assert s1 != s2

    def test_different_class_different_sig(self):
        s1 = compute_signature("KSampler", {"seed": 42}, {})
        s2 = compute_signature("KSamplerAdvanced", {"seed": 42}, {})
        assert s1 != s2

    def test_upstream_sigs_affect_result(self):
        s1 = compute_signature("VAEDecode", {}, {"model": "aaa"})
        s2 = compute_signature("VAEDecode", {}, {"model": "bbb"})
        assert s1 != s2

    def test_is_changed_affects_result(self):
        s1 = compute_signature("LoadImage", {"path": "a.png"}, {})
        s2 = compute_signature("LoadImage", {"path": "a.png"}, {}, is_changed_result="modified")
        assert s1 != s2

    def test_non_scalar_inputs_ignored(self):
        """Non-scalar inputs (tensors, objects) are captured via upstream_sigs, not directly."""
        s1 = compute_signature("KSampler", {"seed": 42, "model": object()}, {})
        s2 = compute_signature("KSampler", {"seed": 42, "model": object()}, {})
        assert s1 == s2  # objects ignored, only scalars hashed

    def test_signature_is_16_chars(self):
        sig = compute_signature("Test", {}, {})
        assert len(sig) == 16

    def test_input_order_doesnt_matter(self):
        """Inputs are sorted by key, so order shouldn't matter."""
        s1 = compute_signature("Test", {"a": 1, "b": 2}, {})
        s2 = compute_signature("Test", {"b": 2, "a": 1}, {})
        assert s1 == s2
