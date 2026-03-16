"""Tests for unified Stagehand model resolver integration in ModelPaths."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_paths(tmp_path, monkeypatch, unified_base=None):
    """Create a ModelPaths instance with controlled directories."""
    # Prevent the real ~/.serenity/models from interfering
    if unified_base is None:
        unified_base = str(tmp_path / "unified_models")
    monkeypatch.setattr(
        "serenityflow.bridge.model_paths._UNIFIED_BASE", unified_base,
    )
    from serenityflow.bridge.model_paths import ModelPaths
    return ModelPaths(str(tmp_path / "legacy"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUnifiedDirInSearchPaths:
    """Verify ~/.serenity/models/<type> dirs appear FIRST in search paths."""

    def test_checkpoints_has_unified_dir(self, tmp_path, monkeypatch):
        unified_base = str(tmp_path / "unified")
        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        expected = os.path.join(unified_base, "checkpoints")
        assert mp.dirs["checkpoints"][0] == expected

    def test_loras_has_unified_dir(self, tmp_path, monkeypatch):
        unified_base = str(tmp_path / "unified")
        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        expected = os.path.join(unified_base, "loras")
        assert mp.dirs["loras"][0] == expected

    def test_vae_maps_to_vaes(self, tmp_path, monkeypatch):
        unified_base = str(tmp_path / "unified")
        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        expected = os.path.join(unified_base, "vaes")
        assert mp.dirs["vae"][0] == expected

    def test_clip_maps_to_text_encoders(self, tmp_path, monkeypatch):
        unified_base = str(tmp_path / "unified")
        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        expected = os.path.join(unified_base, "text_encoders")
        assert mp.dirs["clip"][0] == expected

    def test_embeddings_has_no_unified_dir(self, tmp_path, monkeypatch):
        """Embeddings has no unified equivalent, so no unified dir prepended."""
        unified_base = str(tmp_path / "unified")
        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        # First path should NOT be from unified base
        assert not mp.dirs["embeddings"][0].startswith(unified_base)


class TestFindFromUnifiedDir:
    """Models in the unified directory are found by find()."""

    def test_find_in_unified_dir_via_walk(self, tmp_path, monkeypatch):
        """When stagehand is not available, find() still picks up files from
        the unified dir because it's prepended to search paths."""
        monkeypatch.setattr("serenityflow.bridge.model_paths._HAS_UNIFIED", False)
        unified_base = str(tmp_path / "unified")
        ckpt_dir = os.path.join(unified_base, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        model_file = os.path.join(ckpt_dir, "my_model.safetensors")
        Path(model_file).touch()

        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        result = mp.find("my_model.safetensors", "checkpoints")
        assert result == model_file


class TestFindFallbackToLegacy:
    """Model NOT in unified dir should be found in legacy ComfyUI dir."""

    def test_fallback_to_legacy(self, tmp_path, monkeypatch):
        monkeypatch.setattr("serenityflow.bridge.model_paths._HAS_UNIFIED", False)
        unified_base = str(tmp_path / "unified")

        # Create model in legacy dir only
        legacy_base = tmp_path / "legacy"
        legacy_ckpt = legacy_base / "models" / "checkpoints"
        legacy_ckpt.mkdir(parents=True)
        model_file = legacy_ckpt / "old_model.safetensors"
        model_file.touch()

        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        result = mp.find("old_model.safetensors", "checkpoints")
        assert result == str(model_file)


class TestListModelsIncludesUnified:
    """list_models() should include entries from the unified directory."""

    def test_list_includes_unified_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("serenityflow.bridge.model_paths._HAS_UNIFIED", False)
        unified_base = str(tmp_path / "unified")
        ckpt_dir = os.path.join(unified_base, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        Path(os.path.join(ckpt_dir, "unified_model.safetensors")).touch()

        # Also a legacy model
        legacy_base = tmp_path / "legacy"
        legacy_ckpt = legacy_base / "models" / "checkpoints"
        legacy_ckpt.mkdir(parents=True)
        (legacy_ckpt / "legacy_model.safetensors").touch()

        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        models = mp.list_models("checkpoints")
        assert "unified_model.safetensors" in models
        assert "legacy_model.safetensors" in models


class TestTypeMapping:
    """Each SerenityFlow folder maps to the correct unified type."""

    def test_all_mappings(self):
        from serenityflow.bridge.model_paths import _UNIFIED_TYPE_MAP

        assert _UNIFIED_TYPE_MAP["checkpoints"] == "checkpoints"
        assert _UNIFIED_TYPE_MAP["diffusion_models"] == "checkpoints"
        assert _UNIFIED_TYPE_MAP["loras"] == "loras"
        assert _UNIFIED_TYPE_MAP["vae"] == "vaes"
        assert _UNIFIED_TYPE_MAP["clip"] == "text_encoders"
        assert _UNIFIED_TYPE_MAP["controlnet"] == "controlnets"
        assert _UNIFIED_TYPE_MAP["clip_vision"] == "text_encoders"
        assert _UNIFIED_TYPE_MAP["upscale_models"] == "upscalers"
        assert _UNIFIED_TYPE_MAP["embeddings"] is None
        assert _UNIFIED_TYPE_MAP["style_models"] is None


class TestStandaloneWithoutStagehand:
    """When stagehand is not installed, everything works exactly as before."""

    def test_find_works_without_stagehand(self, tmp_path, monkeypatch):
        monkeypatch.setattr("serenityflow.bridge.model_paths._HAS_UNIFIED", False)
        unified_base = str(tmp_path / "unified")

        legacy_base = tmp_path / "legacy"
        legacy_ckpt = legacy_base / "models" / "checkpoints"
        legacy_ckpt.mkdir(parents=True)
        model_file = legacy_ckpt / "model.safetensors"
        model_file.touch()

        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        result = mp.find("model.safetensors", "checkpoints")
        assert result == str(model_file)

    def test_list_works_without_stagehand(self, tmp_path, monkeypatch):
        monkeypatch.setattr("serenityflow.bridge.model_paths._HAS_UNIFIED", False)
        unified_base = str(tmp_path / "unified")

        legacy_base = tmp_path / "legacy"
        legacy_ckpt = legacy_base / "models" / "checkpoints"
        legacy_ckpt.mkdir(parents=True)
        (legacy_ckpt / "model.safetensors").touch()

        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        models = mp.list_models("checkpoints")
        assert "model.safetensors" in models

    def test_no_error_when_stagehand_missing(self, tmp_path, monkeypatch):
        """Even if _HAS_UNIFIED is False, no import errors or crashes."""
        monkeypatch.setattr("serenityflow.bridge.model_paths._HAS_UNIFIED", False)
        unified_base = str(tmp_path / "unified")
        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        with pytest.raises(FileNotFoundError):
            mp.find("nonexistent.safetensors", "checkpoints")


class TestResolverCheckedFirst:
    """Unified resolver is checked BEFORE legacy dir walk."""

    def test_resolver_takes_priority(self, tmp_path, monkeypatch):
        """When a model exists in both unified resolver and legacy dir,
        the unified resolver result is returned."""
        unified_base = str(tmp_path / "unified")
        unified_path = Path(unified_base) / "checkpoints" / "model"
        unified_path.mkdir(parents=True)
        unified_file = unified_path / "model.safetensors"
        unified_file.touch()

        # Also create in legacy
        legacy_base = tmp_path / "legacy"
        legacy_ckpt = legacy_base / "models" / "checkpoints"
        legacy_ckpt.mkdir(parents=True)
        legacy_file = legacy_ckpt / "model.safetensors"
        legacy_file.touch()

        # Mock the unified resolver to return the unified path
        mock_resolver = MagicMock()
        mock_resolver.resolve_file.return_value = unified_file

        monkeypatch.setattr("serenityflow.bridge.model_paths._HAS_UNIFIED", True)
        monkeypatch.setattr(
            "serenityflow.bridge.model_paths._get_unified_resolver",
            lambda: mock_resolver,
        )

        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        result = mp.find("model.safetensors", "checkpoints")
        assert result == str(unified_file)
        mock_resolver.resolve_file.assert_called()

    def test_resolver_miss_falls_through(self, tmp_path, monkeypatch):
        """When the unified resolver raises FileNotFoundError, legacy dir
        walk still finds the model."""
        unified_base = str(tmp_path / "unified")

        legacy_base = tmp_path / "legacy"
        legacy_ckpt = legacy_base / "models" / "checkpoints"
        legacy_ckpt.mkdir(parents=True)
        legacy_file = legacy_ckpt / "fallback.safetensors"
        legacy_file.touch()

        mock_resolver = MagicMock()
        mock_resolver.resolve_file.side_effect = FileNotFoundError("not found")

        monkeypatch.setattr("serenityflow.bridge.model_paths._HAS_UNIFIED", True)
        monkeypatch.setattr(
            "serenityflow.bridge.model_paths._get_unified_resolver",
            lambda: mock_resolver,
        )

        mp = _make_model_paths(tmp_path, monkeypatch, unified_base=unified_base)
        result = mp.find("fallback.safetensors", "checkpoints")
        assert result == str(legacy_file)
