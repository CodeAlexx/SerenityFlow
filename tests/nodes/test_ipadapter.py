"""Tests for IP-Adapter nodes."""
from __future__ import annotations

import torch
import pytest


def _make_image(batch=1, h=64, w=64, c=3):
    """Create test image tensor BHWC float32 [0,1]."""
    return torch.rand(batch, h, w, c)


class _MockModel:
    """Mock model with with_options() support."""

    def __init__(self, options=None):
        self._options = options or {}

    def with_options(self, opts):
        merged = dict(self._options)
        merged.update(opts)
        return _MockModel(merged)


class TestIPAdapterModelLoader:
    def test_registration(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("IPAdapterModelLoader")
        node = registry.get("IPAdapterModelLoader")
        assert node.return_types == ("IPADAPTER_MODEL",)

    def test_returns_handle_without_bridge(self):
        from serenityflow.nodes.ipadapter import ipadapter_model_loader
        result, = ipadapter_model_loader("test_model")
        assert isinstance(result, dict)
        assert result["model_name"] == "test_model"
        assert result["type"] == "ipadapter"


class TestIPAdapterApplyFull:
    def test_stores_config_via_with_options(self):
        from serenityflow.nodes.ipadapter import ipadapter_apply_full
        model = _MockModel()
        ipadapter = {"model_name": "test", "type": "ipadapter"}
        clip_vision = {"type": "clip_vision"}
        image = _make_image()
        result, = ipadapter_apply_full(
            model, ipadapter, clip_vision, image,
            weight=0.8, weight_type="ease_in",
            start_at=0.1, end_at=0.9, unfold_batch=True,
        )
        assert "ipadapter" in result._options
        cfg = result._options["ipadapter"]
        assert cfg["model"] is ipadapter
        assert cfg["clip_vision"] is clip_vision
        assert cfg["weight"] == 0.8
        assert cfg["weight_type"] == "ease_in"
        assert cfg["start_at"] == 0.1
        assert cfg["end_at"] == 0.9
        assert cfg["unfold_batch"] is True

    def test_registration(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("IPAdapterApplyFull")
        node = registry.get("IPAdapterApplyFull")
        assert node.return_types == ("MODEL",)

    def test_passthrough_without_with_options(self):
        from serenityflow.nodes.ipadapter import ipadapter_apply_full
        model = "plain_model"  # no with_options
        result, = ipadapter_apply_full(
            model, {}, {}, _make_image(),
        )
        assert result == "plain_model"


class TestIPAdapterApplyFaceID:
    def test_stores_face_id_flag(self):
        from serenityflow.nodes.ipadapter import ipadapter_apply_face_id
        model = _MockModel()
        ipadapter = {"model_name": "faceid", "type": "ipadapter"}
        clip_vision = {"type": "clip_vision"}
        image = _make_image()
        result, = ipadapter_apply_face_id(
            model, ipadapter, clip_vision, image,
            weight=0.7, start_at=0.0, end_at=1.0,
        )
        cfg = result._options["ipadapter"]
        assert cfg["face_id"] is True
        assert cfg["weight"] == 0.7
        assert cfg["model"] is ipadapter

    def test_registration(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("IPAdapterApplyFaceID")
        node = registry.get("IPAdapterApplyFaceID")
        assert node.return_types == ("MODEL",)
        assert node.category == "conditioning/ipadapter"

    def test_passthrough_without_with_options(self):
        from serenityflow.nodes.ipadapter import ipadapter_apply_face_id
        model = 42  # no with_options
        result, = ipadapter_apply_face_id(model, {}, {}, _make_image())
        assert result == 42
