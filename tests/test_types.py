from __future__ import annotations
import torch
import pytest
from serenityflow.core.types import (
    ModelHandle, ClipHandle, VaeHandle, ControlNetHandle,
    validate_image, validate_mask, validate_latent, validate_wire,
    _deep_merge, WIRE_TYPES,
)
from serenityflow.core.patch_ledger import PatchLedger, PatchEntry


def _make_handle(path="model.safetensors", ledger=None):
    import uuid
    if ledger is None:
        ledger = PatchLedger()
    return ModelHandle(
        handle_id=uuid.uuid4().hex,
        arch="flux",
        config={"layers": 24},
        path=path,
        dtype=torch.bfloat16,
        patch_ledger=ledger,
    )


class TestModelHandle:
    def test_frozen(self):
        h = _make_handle()
        with pytest.raises(AttributeError):
            h.arch = "sdxl"

    def test_with_patches_new_id_and_cache_key(self):
        h = _make_handle()
        ledger2 = PatchLedger()
        ledger2.add_patch(PatchEntry("lora", "x.safetensors", 1.0, {"k"}, {}))
        h2 = h.with_patches(ledger2)
        assert h2.handle_id != h.handle_id
        assert h2.cache_key() != h.cache_key()

    def test_with_options_new_id_and_cache_key(self):
        h = _make_handle()
        h2 = h.with_options({"sampling": {"cfg": 7.5}})
        assert h2.handle_id != h.handle_id
        assert h2.model_options == {"sampling": {"cfg": 7.5}}

    def test_two_handles_different_ids(self):
        h1 = _make_handle()
        h2 = _make_handle()
        assert h1.handle_id != h2.handle_id


class TestDeepMerge:
    def test_nested(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10, "e": 5}, "f": 6}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": 10, "c": 2, "e": 5}, "d": 3, "f": 6}

    def test_override_non_dict(self):
        result = _deep_merge({"a": 1}, {"a": {"nested": True}})
        assert result == {"a": {"nested": True}}


class TestValidateImage:
    def test_valid(self):
        t = torch.rand(1, 64, 64, 3)
        assert validate_image(t) == []

    def test_valid_single_channel(self):
        t = torch.rand(1, 64, 64, 1)
        assert validate_image(t) == []

    def test_wrong_dims(self):
        t = torch.rand(64, 64, 3)
        errs = validate_image(t)
        assert any("4D" in e for e in errs)

    def test_wrong_dtype(self):
        t = torch.rand(1, 64, 64, 3).half()
        errs = validate_image(t)
        assert any("float32" in e for e in errs)

    def test_wrong_range(self):
        t = torch.ones(1, 64, 64, 3) * 2.0
        errs = validate_image(t)
        assert any("range" in e for e in errs)

    def test_not_tensor(self):
        errs = validate_image("nope")
        assert any("Tensor" in e for e in errs)


class TestValidateMask:
    def test_valid_3d(self):
        t = torch.rand(1, 64, 64)
        assert validate_mask(t) == []

    def test_valid_2d(self):
        t = torch.rand(64, 64)
        assert validate_mask(t) == []

    def test_wrong_dtype(self):
        t = torch.rand(64, 64).half()
        errs = validate_mask(t)
        assert any("float32" in e for e in errs)


class TestValidateLatent:
    def test_valid(self):
        d = {"samples": torch.randn(1, 4, 64, 64)}
        assert validate_latent(d) == []

    def test_missing_samples(self):
        errs = validate_latent({"noise": torch.randn(1, 4, 64, 64)})
        assert any("samples" in e for e in errs)

    def test_not_dict(self):
        errs = validate_latent("nope")
        assert any("dict" in e for e in errs)


class TestValidateWire:
    def test_unknown_type_passes(self):
        assert validate_wire("STRING", "hello") == []
        assert validate_wire("NONEXISTENT", object()) == []

    def test_delegates_to_validator(self):
        errs = validate_wire("IMAGE", torch.rand(64, 64))
        assert len(errs) > 0
