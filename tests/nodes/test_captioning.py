"""Tests for captioning nodes -- JoyCaption and WD14 tagger."""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image():
    """1x32x32x3 float32 image."""
    return torch.rand(1, 32, 32, 3)


# ---------------------------------------------------------------------------
# JoyCaptionModelLoader
# ---------------------------------------------------------------------------

class TestJoyCaptionModelLoader:
    def test_registered(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("JoyCaptionModelLoader")

    def test_returns_handle(self):
        from serenityflow.nodes.captioning import joycaption_model_loader
        (model,) = joycaption_model_loader("test-model")
        assert isinstance(model, dict)
        assert model["_type"] == "joycaption"
        assert model["model_name"] == "test-model"

    def test_return_types(self):
        from serenityflow.nodes.registry import registry
        node = registry.get("JoyCaptionModelLoader")
        assert node.return_types == ("JOYCAPTION_MODEL",)


# ---------------------------------------------------------------------------
# JoyCaptionAdvanced
# ---------------------------------------------------------------------------

class TestJoyCaptionAdvanced:
    def test_registered(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("JoyCaptionAdvanced")

    def test_with_mock_model(self, dummy_image):
        """Mock model with .caption() method returns a string."""
        from serenityflow.nodes.captioning import joycaption_advanced

        class MockModel:
            def caption(self, image, mode="descriptive", length="medium", focus_on="general"):
                return f"A {mode} caption of length {length}"

        (result,) = joycaption_advanced(MockModel(), dummy_image, mode="art_critic", length="short")
        assert isinstance(result, str)
        assert "art_critic" in result
        assert "short" in result

    def test_handle_without_bridge_raises(self, dummy_image):
        """Lazy handle without bridge should raise NotImplementedError."""
        from serenityflow.nodes.captioning import joycaption_advanced
        handle = {"_type": "joycaption", "model_name": "test"}
        with pytest.raises(NotImplementedError):
            joycaption_advanced(handle, dummy_image)

    def test_default_params(self):
        from serenityflow.nodes.registry import registry
        node = registry.get("JoyCaptionAdvanced")
        opt = node.input_types.get("optional", {})
        assert "mode" in opt
        assert "length" in opt
        assert "focus_on" in opt


# ---------------------------------------------------------------------------
# WD14ModelLoader
# ---------------------------------------------------------------------------

class TestWD14ModelLoader:
    def test_registered(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("WD14ModelLoader")

    def test_returns_handle(self):
        from serenityflow.nodes.captioning import wd14_model_loader
        (model,) = wd14_model_loader("wd-v1-4-moat-tagger-v2")
        assert isinstance(model, dict)
        assert model["_type"] == "wd14"

    def test_return_types(self):
        from serenityflow.nodes.registry import registry
        node = registry.get("WD14ModelLoader")
        assert node.return_types == ("WD14_MODEL",)


# ---------------------------------------------------------------------------
# WD14Tag
# ---------------------------------------------------------------------------

class TestWD14Tag:
    def test_registered(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("WD14Tag")

    def test_with_mock_model_string(self, dummy_image):
        """Mock model returning a comma-separated string."""
        from serenityflow.nodes.captioning import wd14_tag

        class MockTagger:
            def tag(self, image, threshold=0.35, character_threshold=0.85):
                return "1girl, solo, smile, blue_hair"

        (result,) = wd14_tag(MockTagger(), dummy_image)
        assert isinstance(result, str)
        assert "1girl" in result

    def test_with_mock_model_list(self, dummy_image):
        """Mock model returning a list -- should be joined."""
        from serenityflow.nodes.captioning import wd14_tag

        class MockTagger:
            def tag(self, image, threshold=0.35, character_threshold=0.85):
                return ["1girl", "solo", "smile"]

        (result,) = wd14_tag(MockTagger(), dummy_image)
        assert result == "1girl, solo, smile"

    def test_handle_without_bridge_raises(self, dummy_image):
        from serenityflow.nodes.captioning import wd14_tag
        handle = {"_type": "wd14", "model_name": "test"}
        with pytest.raises(NotImplementedError):
            wd14_tag(handle, dummy_image)

    def test_threshold_params(self):
        from serenityflow.nodes.registry import registry
        node = registry.get("WD14Tag")
        opt = node.input_types.get("optional", {})
        assert "threshold" in opt
        assert "character_threshold" in opt
