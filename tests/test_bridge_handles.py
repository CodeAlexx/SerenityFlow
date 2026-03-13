"""Tests for bridge handle integration with core types."""
from __future__ import annotations

import uuid
import pytest
import torch

from serenityflow.core.types import ModelHandle, ClipHandle, VaeHandle, ControlNetHandle
from serenityflow.core.patch_ledger import PatchLedger, PatchEntry


def _make_model(**kwargs):
    defaults = dict(
        handle_id=uuid.uuid4().hex,
        arch="flux",
        config={"double_blocks": 19, "single_blocks": 38},
        path="/models/flux1-dev.safetensors",
        dtype=torch.bfloat16,
        patch_ledger=PatchLedger(),
    )
    defaults.update(kwargs)
    return ModelHandle(**defaults)


class TestModelHandleCreation:
    def test_basic_creation(self):
        m = _make_model()
        assert m.arch == "flux"
        assert m.path == "/models/flux1-dev.safetensors"
        assert m.dtype == torch.bfloat16
        assert isinstance(m.handle_id, str)
        assert len(m.handle_id) == 32  # uuid hex

    def test_has_patch_ledger(self):
        m = _make_model()
        assert isinstance(m.patch_ledger, PatchLedger)
        assert m.patch_ledger.epoch == 0

    def test_default_model_options(self):
        m = _make_model()
        assert m.model_options == {}


class TestModelHandlePatches:
    def test_with_patches_new_id(self):
        m = _make_model()
        ledger = PatchLedger()
        ledger.add_patch(PatchEntry("lora", "style.safetensors", 0.8, {"k"}, {}))
        m2 = m.with_patches(ledger)
        assert m2.handle_id != m.handle_id
        assert m2.patch_ledger is ledger

    def test_cache_key_changes_with_patches(self):
        m = _make_model()
        ck1 = m.cache_key()
        ledger = PatchLedger()
        ledger.add_patch(PatchEntry("lora", "style.safetensors", 0.8, {"k"}, {}))
        m2 = m.with_patches(ledger)
        ck2 = m2.cache_key()
        assert ck1 != ck2

    def test_original_unchanged(self):
        m = _make_model()
        original_id = m.handle_id
        ledger = PatchLedger()
        ledger.add_patch(PatchEntry("lora", "x.safetensors", 1.0, {"k"}, {}))
        _ = m.with_patches(ledger)
        assert m.handle_id == original_id
        assert m.patch_ledger.epoch == 0


class TestModelHandleOptions:
    def test_with_options_new_id(self):
        m = _make_model()
        m2 = m.with_options({"transformer_options": {"patch": True}})
        assert m2.handle_id != m.handle_id

    def test_options_deep_merge(self):
        m = _make_model(model_options={"transformer_options": {"a": 1}})
        m2 = m.with_options({"transformer_options": {"b": 2}})
        assert m2.model_options == {"transformer_options": {"a": 1, "b": 2}}


class TestClipHandle:
    def test_creation(self):
        c = ClipHandle(handle_id=uuid.uuid4().hex, path="/models/clip_l.safetensors")
        assert isinstance(c.handle_id, str)
        assert c.cache_key() == c.handle_id

    def test_frozen(self):
        c = ClipHandle(handle_id="abc", path="/models/clip.safetensors")
        with pytest.raises(AttributeError):
            c.path = "other"


class TestVaeHandle:
    def test_creation(self):
        v = VaeHandle(handle_id=uuid.uuid4().hex, path="/models/vae.safetensors")
        assert isinstance(v.handle_id, str)
        assert v.cache_key() == v.handle_id


class TestControlNetHandle:
    def test_creation(self):
        cn = ControlNetHandle(handle_id=uuid.uuid4().hex, path="/models/cn.safetensors")
        assert isinstance(cn.handle_id, str)
        assert cn.cache_key() == cn.handle_id


class TestCacheKeyBehavior:
    def test_same_ledger_same_cache_key(self):
        """Two handles with same ledger state should produce same fingerprint component."""
        ledger = PatchLedger()
        m1 = _make_model(patch_ledger=ledger)
        # Same handle_id would give same cache_key
        m2 = ModelHandle(
            handle_id=m1.handle_id,
            arch=m1.arch, config=m1.config, path=m1.path,
            dtype=m1.dtype, patch_ledger=ledger,
        )
        assert m1.cache_key() == m2.cache_key()

    def test_different_strength_different_fingerprint(self):
        l1 = PatchLedger()
        l1.add_patch(PatchEntry("lora", "x.safetensors", 0.5, {"k"}, {}))
        l2 = PatchLedger()
        l2.add_patch(PatchEntry("lora", "x.safetensors", 1.0, {"k"}, {}))
        assert l1.fingerprint() != l2.fingerprint()
