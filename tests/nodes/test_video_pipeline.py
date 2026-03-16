"""Tests for video_pipeline nodes -- I/O, latent ops, temporal VAE, interpolation."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def frames_4():
    """4 video frames, 32x32 RGB."""
    return torch.rand(4, 32, 32, 3)


@pytest.fixture
def frames_8():
    """8 video frames, 16x16 RGB."""
    return torch.rand(8, 16, 16, 3)


@pytest.fixture
def latent_4():
    """Latent batch of 4 frames."""
    return {"samples": torch.randn(4, 4, 8, 8)}


@pytest.fixture
def latent_8():
    """Latent batch of 8 frames."""
    return {"samples": torch.randn(8, 4, 8, 8)}


@pytest.fixture
def latent_1():
    """Single-frame latent."""
    return {"samples": torch.randn(1, 4, 8, 8)}


# ---------------------------------------------------------------------------
# LoadVideoFrames
# ---------------------------------------------------------------------------

class TestLoadVideoFrames:
    def test_load_png_frames(self):
        """Load 5 PNG frames from temp dir, verify batch shape."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                img = Image.fromarray(
                    (torch.rand(24, 32, 3).numpy() * 255).astype("uint8")
                )
                img.save(os.path.join(d, f"frame_{i:03d}.png"))

            fn = registry.get_function("LoadVideoFrames")
            frames, count = fn(directory=d, pattern="*.png")

            assert frames.shape == (5, 24, 32, 3)
            assert count == 5
            assert frames.dtype == torch.float32
            assert frames.min() >= 0.0
            assert frames.max() <= 1.0

    def test_missing_dir_raises(self):
        fn = registry.get_function("LoadVideoFrames")
        with pytest.raises(FileNotFoundError):
            fn(directory="/nonexistent/path")

    def test_no_matching_files_raises(self):
        fn = registry.get_function("LoadVideoFrames")
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError, match="No files matching"):
                fn(directory=d, pattern="*.png")


# ---------------------------------------------------------------------------
# SaveVideoFrames
# ---------------------------------------------------------------------------

class TestSaveVideoFrames:
    def test_save_frames(self, frames_4):
        fn = registry.get_function("SaveVideoFrames")
        with tempfile.TemporaryDirectory() as d:
            fn(images=frames_4, output_dir=d, prefix="test_")
            files = sorted(os.listdir(d))
            assert len(files) == 4
            assert files[0] == "test_00000.png"
            assert files[3] == "test_00003.png"


# ---------------------------------------------------------------------------
# MergeVideoFrames
# ---------------------------------------------------------------------------

class TestMergeVideoFrames:
    def test_merge_same_size(self, frames_4):
        fn = registry.get_function("MergeVideoFrames")
        other = torch.rand(3, 32, 32, 3)
        (merged,) = fn(frames_1=frames_4, frames_2=other)
        assert merged.shape == (7, 32, 32, 3)

    def test_merge_different_size(self, frames_4):
        """frames_2 spatial dims differ -- should be resized to match frames_1."""
        fn = registry.get_function("MergeVideoFrames")
        other = torch.rand(2, 64, 64, 3)
        (merged,) = fn(frames_1=frames_4, frames_2=other)
        assert merged.shape == (6, 32, 32, 3)


# ---------------------------------------------------------------------------
# SplitVideoFrames
# ---------------------------------------------------------------------------

class TestSplitVideoFrames:
    def test_split_first(self, frames_4):
        fn = registry.get_function("SplitVideoFrames")
        (first,) = fn(images=frames_4)
        assert first.shape == (1, 32, 32, 3)
        assert torch.allclose(first[0], frames_4[0])


# ---------------------------------------------------------------------------
# SetLatentBatchSize
# ---------------------------------------------------------------------------

class TestSetLatentBatchSize:
    def test_expand(self, latent_1):
        fn = registry.get_function("SetLatentBatchSize")
        (result,) = fn(samples=latent_1, batch_size=4)
        assert result["samples"].shape[0] == 4

    def test_truncate(self, latent_8):
        fn = registry.get_function("SetLatentBatchSize")
        (result,) = fn(samples=latent_8, batch_size=3)
        assert result["samples"].shape[0] == 3

    def test_noop(self, latent_4):
        fn = registry.get_function("SetLatentBatchSize")
        (result,) = fn(samples=latent_4, batch_size=4)
        assert result["samples"].shape[0] == 4


# ---------------------------------------------------------------------------
# LatentBatchSlice
# ---------------------------------------------------------------------------

class TestLatentBatchSlice:
    def test_slice_middle(self, latent_8):
        fn = registry.get_function("LatentBatchSlice")
        (result,) = fn(samples=latent_8, start=2, end=5)
        assert result["samples"].shape[0] == 3

    def test_slice_to_end(self, latent_8):
        fn = registry.get_function("LatentBatchSlice")
        (result,) = fn(samples=latent_8, start=5, end=-1)
        assert result["samples"].shape[0] == 3

    def test_slice_from_start(self, latent_8):
        fn = registry.get_function("LatentBatchSlice")
        (result,) = fn(samples=latent_8, start=0, end=2)
        assert result["samples"].shape[0] == 2


# ---------------------------------------------------------------------------
# AnimateDiff Nodes
# ---------------------------------------------------------------------------

class TestAnimateDiff:
    def test_loader_registration(self):
        node = registry.get("AnimateDiffLoader")
        assert node is not None
        assert "ANIMATEDIFF_MODEL" in node.return_types

    def test_settings_structure(self):
        fn = registry.get_function("AnimateDiffSettings")
        (settings,) = fn(motion_scale=1.5, beta_schedule="cosine")
        assert isinstance(settings, dict)
        assert settings["motion_scale"] == 1.5
        assert settings["beta_schedule"] == "cosine"

    def test_loader_returns_handle(self):
        fn = registry.get_function("AnimateDiffLoader")
        (model,) = fn(model_name="v3_adapter_sd_v15.ckpt")
        assert isinstance(model, dict)
        assert model["type"] == "animatediff"

    def test_combine_loop(self, frames_4):
        fn = registry.get_function("AnimateDiffCombine")
        (result,) = fn(images=frames_4, loop=True)
        # 4 original + 2 reversed (indices 1,2 reversed) = 6
        assert result.shape[0] == 6

    def test_combine_no_loop(self, frames_4):
        fn = registry.get_function("AnimateDiffCombine")
        (result,) = fn(images=frames_4, loop=False)
        assert result.shape[0] == 4


# ---------------------------------------------------------------------------
# WanT2V / WanV2V
# ---------------------------------------------------------------------------

class TestWanNodes:
    def test_wan_t2v_registration(self):
        node = registry.get("WanT2V")
        assert node is not None
        req = node.input_types["required"]
        assert "model" in req
        assert "positive" in req
        assert "frames" in req

    def test_wan_v2v_registration(self):
        node = registry.get("WanV2V")
        assert node is not None
        req = node.input_types["required"]
        assert "video" in req
        opt = node.input_types.get("optional", {})
        assert "denoise" in opt

    def test_wan_t2v_raises(self):
        fn = registry.get_function("WanT2V")
        with pytest.raises(NotImplementedError):
            fn(model=None, positive=None, negative=None)

    def test_wan_v2v_raises(self):
        fn = registry.get_function("WanV2V")
        with pytest.raises(NotImplementedError):
            fn(model=None, positive=None, negative=None, video=None)


# ---------------------------------------------------------------------------
# VAEDecodeVideo / VAEEncodeVideo
# ---------------------------------------------------------------------------

class TestTemporalVAE:
    def test_decode_chunking(self):
        """Mock VAE, verify temporal chunking calls correct number of times."""
        vae = MagicMock()
        # Return BCHW tensor from decode
        vae.decode.side_effect = lambda x: torch.rand(x.shape[0], 3, 32, 32)

        latent = {"samples": torch.randn(12, 4, 8, 8)}
        fn = registry.get_function("VAEDecodeVideo")
        (result,) = fn(vae=vae, samples=latent, tile_temporal=4)

        # 12 frames / 4 chunk = 3 calls
        assert vae.decode.call_count == 3
        # Result is BHWC
        assert result.shape == (12, 32, 32, 3)

    def test_decode_non_divisible(self):
        """Non-divisible frame count handled correctly."""
        vae = MagicMock()
        vae.decode.side_effect = lambda x: torch.rand(x.shape[0], 3, 16, 16)

        latent = {"samples": torch.randn(7, 4, 4, 4)}
        fn = registry.get_function("VAEDecodeVideo")
        (result,) = fn(vae=vae, samples=latent, tile_temporal=3)

        # 7 frames: chunks of [3, 3, 1] = 3 calls
        assert vae.decode.call_count == 3
        assert result.shape[0] == 7

    def test_encode_chunking(self):
        """Mock VAE, verify temporal chunking for encode."""
        vae = MagicMock()
        vae.encode.side_effect = lambda x: torch.randn(x.shape[0], 4, 4, 4)

        images = torch.rand(10, 32, 32, 3)  # BHWC
        fn = registry.get_function("VAEEncodeVideo")
        (result,) = fn(vae=vae, images=images, tile_temporal=4)

        # 10 frames / 4 chunk = 3 calls (4+4+2)
        assert vae.encode.call_count == 3
        assert result["samples"].shape[0] == 10

    def test_encode_single_chunk(self):
        """All frames fit in one chunk."""
        vae = MagicMock()
        vae.encode.side_effect = lambda x: torch.randn(x.shape[0], 4, 8, 8)

        images = torch.rand(3, 16, 16, 3)
        fn = registry.get_function("VAEEncodeVideo")
        (result,) = fn(vae=vae, images=images, tile_temporal=8)

        assert vae.encode.call_count == 1
        assert result["samples"].shape[0] == 3


# ---------------------------------------------------------------------------
# RIFEInterpolate
# ---------------------------------------------------------------------------

class TestRIFEInterpolate:
    def test_linear_fallback_2x(self, frames_4):
        """4 frames with 2x = 7 output frames (linear interpolation fallback)."""
        fn = registry.get_function("RIFEInterpolate")
        rife_model = {"model_name": "test", "type": "rife"}
        (result,) = fn(rife_model=rife_model, images=frames_4, multiplier=2)

        # 4 original + 3 interpolated = 7
        assert result.shape[0] == 7
        assert result.shape[1:] == frames_4.shape[1:]

    def test_linear_fallback_4x(self, frames_4):
        """4 frames with 4x = 13 output frames."""
        fn = registry.get_function("RIFEInterpolate")
        rife_model = {"model_name": "test", "type": "rife"}
        (result,) = fn(rife_model=rife_model, images=frames_4, multiplier=4)

        # 4 original + 3*3 interpolated = 13
        assert result.shape[0] == 13

    def test_single_frame_passthrough(self):
        """Single frame returns unchanged."""
        fn = registry.get_function("RIFEInterpolate")
        single = torch.rand(1, 16, 16, 3)
        (result,) = fn(rife_model={}, images=single, multiplier=2)
        assert result.shape[0] == 1

    def test_interpolated_values(self):
        """Check that interpolated frame is between source frames."""
        fn = registry.get_function("RIFEInterpolate")
        black = torch.zeros(1, 4, 4, 3)
        white = torch.ones(1, 4, 4, 3)
        frames = torch.cat([black, white], dim=0)

        (result,) = fn(rife_model={}, images=frames, multiplier=2)
        # Middle frame should be ~0.5
        assert result.shape[0] == 3
        mid = result[1]
        assert torch.allclose(mid, torch.full_like(mid, 0.5), atol=0.01)


# ---------------------------------------------------------------------------
# GetVideoInfo (skip if no ffprobe/imageio)
# ---------------------------------------------------------------------------

class TestGetVideoInfo:
    def test_missing_file_raises(self):
        fn = registry.get_function("GetVideoInfo")
        with pytest.raises(FileNotFoundError):
            fn(video_path="/nonexistent/video.mp4")

    def test_registration(self):
        node = registry.get("GetVideoInfo")
        assert node is not None
        assert len(node.return_types) == 5
        assert "INT" in node.return_types
        assert "FLOAT" in node.return_types


# ---------------------------------------------------------------------------
# Registration Validation
# ---------------------------------------------------------------------------

class TestRegistrations:
    @pytest.mark.parametrize("name", [
        "LoadVideo",
        "LoadVideoFrames",
        "SaveVideoFrames",
        "CombineVideoFrames",
        "SplitVideoFrames",
        "MergeVideoFrames",
        "GetVideoInfo",
        "AnimateDiffLoader",
        "AnimateDiffSettings",
        "AnimateDiffCombine",
        "SetLatentBatchSize",
        "LatentBatchSlice",
        "WanT2V",
        "WanV2V",
        "LTXAudioSync",
        "VAEDecodeVideo",
        "VAEEncodeVideo",
        "LoadRIFEModel",
        "RIFEInterpolate",
    ])
    def test_node_registered(self, name):
        assert registry.has(name), f"{name} not registered"
        node = registry.get(name)
        assert node.fn is not None
        assert len(node.return_types) >= 0
