"""Tests for face workflow nodes -- SAM3, Florence2, face detection,
inpaint crop/stitch, FaceDetailer, mask combine, captioning aliases."""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def img_64():
    """64x64 RGB image, batch=1, random values."""
    return torch.rand(1, 64, 64, 3)


@pytest.fixture
def img_128():
    """128x128 RGB image, batch=1, random values."""
    return torch.rand(1, 128, 128, 3)


@pytest.fixture
def mask_64():
    """64x64 mask, batch=1, all zeros."""
    return torch.zeros(1, 64, 64)


@pytest.fixture
def mask_64_center():
    """64x64 mask with a 16x16 white square in the center."""
    m = torch.zeros(1, 64, 64)
    m[:, 24:40, 24:40] = 1.0
    return m


@pytest.fixture
def mask_128_center():
    """128x128 mask with a 32x32 white square in the center."""
    m = torch.zeros(1, 128, 128)
    m[:, 48:80, 48:80] = 1.0
    return m


# ---------------------------------------------------------------------------
# Registration checks
# ---------------------------------------------------------------------------

class TestRegistration:
    """Verify all face workflow nodes are registered."""

    @pytest.mark.parametrize("node_name", [
        "LoadSAM3Model",
        "SAM3Grounding",
        "Florence2ModelLoader",
        "Florence2Run",
        "Florence2Caption",
        "UltralyticsDetectorLoader",
        "UltralyticsDetectorRun",
        "FaceRestoreModelLoader",
        "FaceRestore",
        "InpaintCropImproved",
        "InpaintStitch",
        "FaceDetailer",
        "JoyCaptionLoad",
        "MasksCombineRegions",
    ])
    def test_node_registered(self, node_name):
        from serenityflow.nodes.registry import registry
        assert registry.has(node_name), f"{node_name} not registered"

    def test_face_detailer_input_types(self):
        from serenityflow.nodes.registry import registry
        node = registry.get("FaceDetailer")
        req = node.input_types["required"]
        assert "image" in req
        assert "model" in req
        assert "clip" in req
        assert "vae" in req
        assert "positive" in req
        assert "negative" in req

    def test_face_detailer_output_types(self):
        from serenityflow.nodes.registry import registry
        node = registry.get("FaceDetailer")
        assert node.return_types == ("IMAGE", "MASK")


# ---------------------------------------------------------------------------
# SAM3 / Florence2 fallback behavior
# ---------------------------------------------------------------------------

class TestSAM3Fallback:
    def test_grounding_fallback_returns_empty_mask(self, img_64):
        from serenityflow.nodes.face_workflow import sam3_grounding
        handle = {"_type": "sam3", "model_name": "test"}
        mask, bboxes = sam3_grounding(handle, img_64, "face")
        assert mask.shape == (1, 64, 64)
        assert mask.sum().item() == 0.0
        assert bboxes == []

    def test_grounding_fallback_preserves_batch(self):
        from serenityflow.nodes.face_workflow import sam3_grounding
        img = torch.rand(3, 32, 48, 3)
        handle = {"_type": "sam3", "model_name": "test"}
        mask, bboxes = sam3_grounding(handle, img, "person")
        assert mask.shape == (3, 32, 48)


class TestFlorence2Fallback:
    def test_run_fallback_returns_empty(self, img_64):
        from serenityflow.nodes.face_workflow import florence2_run
        handle = {"_type": "florence2", "model_name": "test"}
        text, mask, bboxes = florence2_run(handle, img_64, task="caption")
        assert text == ""
        assert mask.shape == (1, 64, 64)
        assert bboxes == []

    def test_caption_fallback_returns_string(self, img_64):
        from serenityflow.nodes.face_workflow import florence2_caption
        handle = {"_type": "florence2", "model_name": "test"}
        (text,) = florence2_caption(handle, img_64, detail_level="brief")
        assert isinstance(text, str)
        assert text == ""


# ---------------------------------------------------------------------------
# Detection fallback
# ---------------------------------------------------------------------------

class TestDetectionFallback:
    def test_ultralytics_fallback(self, img_64):
        from serenityflow.nodes.face_workflow import ultralytics_detector_run
        handle = {"_type": "ultralytics", "model_name": "test"}
        bboxes, mask = ultralytics_detector_run(handle, img_64)
        assert bboxes == []
        assert mask.shape == (1, 64, 64)

    def test_face_restore_fallback_passthrough(self, img_64):
        from serenityflow.nodes.face_workflow import face_restore
        handle = {"_type": "face_restore", "model_name": "test"}
        (result,) = face_restore(handle, img_64, fidelity=0.5)
        assert torch.equal(result, img_64)


# ---------------------------------------------------------------------------
# InpaintCropImproved
# ---------------------------------------------------------------------------

class TestInpaintCropImproved:
    def test_crop_around_center_mask(self, img_128, mask_128_center):
        from serenityflow.nodes.face_workflow import inpaint_crop_improved
        cropped, cropped_mask, stitch = inpaint_crop_improved(
            img_128, mask_128_center, min_size=32, rescale_factor=1.0, padding=8,
        )
        # Cropped should be smaller than original
        assert cropped.shape[1] < 128 or cropped.shape[2] < 128
        assert cropped.shape[0] == 1
        assert cropped.shape[3] == 3
        # Mask same spatial dims as cropped image
        assert cropped_mask.shape[1] == cropped.shape[1]
        assert cropped_mask.shape[2] == cropped.shape[2]

    def test_stitch_data_has_required_keys(self, img_128, mask_128_center):
        from serenityflow.nodes.face_workflow import inpaint_crop_improved
        _, _, stitch = inpaint_crop_improved(
            img_128, mask_128_center, min_size=32, padding=8,
        )
        for key in ("x", "y", "crop_w", "crop_h", "orig_w", "orig_h",
                     "rescale_factor", "scaled_w", "scaled_h"):
            assert key in stitch, f"Missing key: {key}"

    def test_empty_mask_returns_full_image(self, img_64, mask_64):
        from serenityflow.nodes.face_workflow import inpaint_crop_improved
        cropped, cropped_mask, stitch = inpaint_crop_improved(
            img_64, mask_64, min_size=32, padding=8,
        )
        assert cropped.shape == img_64.shape

    def test_min_size_enforcement(self, img_128):
        from serenityflow.nodes.face_workflow import inpaint_crop_improved
        # Small 4x4 mask region
        m = torch.zeros(1, 128, 128)
        m[:, 60:64, 60:64] = 1.0
        cropped, _, stitch = inpaint_crop_improved(
            img_128, m, min_size=64, padding=0,
        )
        # Crop should be at least min_size in both dims
        assert cropped.shape[1] >= 64 or cropped.shape[1] == 128
        assert cropped.shape[2] >= 64 or cropped.shape[2] == 128

    def test_rescale_factor(self, img_128, mask_128_center):
        from serenityflow.nodes.face_workflow import inpaint_crop_improved
        cropped_1x, _, stitch_1x = inpaint_crop_improved(
            img_128, mask_128_center, min_size=32, rescale_factor=1.0, padding=8,
        )
        cropped_2x, _, stitch_2x = inpaint_crop_improved(
            img_128, mask_128_center, min_size=32, rescale_factor=2.0, padding=8,
        )
        # 2x rescale should produce larger crop
        assert cropped_2x.shape[1] > cropped_1x.shape[1]
        assert cropped_2x.shape[2] > cropped_1x.shape[2]
        # Stitch data should still reference original crop size
        assert stitch_2x["crop_h"] == stitch_1x["crop_h"]
        assert stitch_2x["crop_w"] == stitch_1x["crop_w"]


# ---------------------------------------------------------------------------
# InpaintStitch
# ---------------------------------------------------------------------------

class TestInpaintStitch:
    def test_roundtrip_preserves_outside_mask(self, img_128, mask_128_center):
        """Crop + stitch should preserve pixels outside the mask region."""
        from serenityflow.nodes.face_workflow import inpaint_crop_improved, inpaint_stitch

        cropped, _, stitch = inpaint_crop_improved(
            img_128, mask_128_center, min_size=32, padding=8,
        )
        # Stitch back (using the same cropped content)
        (result,) = inpaint_stitch(img_128, cropped, stitch)
        assert result.shape == img_128.shape

        # Pixels far from the mask (corners) should be very close to original
        # Top-left 8x8 block
        corner_orig = img_128[:, :8, :8, :]
        corner_result = result[:, :8, :8, :]
        assert torch.allclose(corner_orig, corner_result, atol=1e-5)

    def test_output_clamped(self, img_128, mask_128_center):
        from serenityflow.nodes.face_workflow import inpaint_crop_improved, inpaint_stitch
        cropped, _, stitch = inpaint_crop_improved(
            img_128, mask_128_center, min_size=32, padding=8,
        )
        # Use a bright crop to test clamping
        bright_crop = torch.ones_like(cropped) * 1.5
        (result,) = inpaint_stitch(img_128, bright_crop, stitch)
        assert result.max() <= 1.0
        assert result.min() >= 0.0


# ---------------------------------------------------------------------------
# MasksCombineRegions
# ---------------------------------------------------------------------------

class TestMasksCombineRegions:
    def test_union_non_overlapping(self):
        from serenityflow.nodes.face_workflow import masks_combine_regions
        m1 = torch.zeros(1, 32, 32)
        m1[:, :16, :] = 1.0  # top half
        m2 = torch.zeros(1, 32, 32)
        m2[:, 16:, :] = 1.0  # bottom half
        (result,) = masks_combine_regions(m1, m2)
        # Union should be all ones
        assert torch.allclose(result, torch.ones(1, 32, 32))

    def test_union_with_optional_masks(self):
        from serenityflow.nodes.face_workflow import masks_combine_regions
        m1 = torch.zeros(1, 16, 16)
        m1[:, :4, :] = 1.0
        m2 = torch.zeros(1, 16, 16)
        m2[:, 4:8, :] = 1.0
        m3 = torch.zeros(1, 16, 16)
        m3[:, 8:12, :] = 1.0
        (result,) = masks_combine_regions(m1, m2, mask_3=m3)
        assert result[:, :12, :].sum() > 0
        assert result[:, 12:, :].sum() == 0

    def test_2d_mask_input(self):
        from serenityflow.nodes.face_workflow import masks_combine_regions
        m1 = torch.zeros(16, 16)
        m1[:8, :] = 1.0
        m2 = torch.zeros(16, 16)
        m2[8:, :] = 1.0
        (result,) = masks_combine_regions(m1, m2)
        assert result.ndim == 3
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# Florence2Caption
# ---------------------------------------------------------------------------

class TestFlorence2Caption:
    def test_returns_string(self, img_64):
        from serenityflow.nodes.face_workflow import florence2_caption
        handle = {"_type": "florence2", "model_name": "test"}
        (text,) = florence2_caption(handle, img_64, detail_level="detailed")
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Loader nodes return handle dicts
# ---------------------------------------------------------------------------

class TestLoaders:
    def test_sam3_loader(self):
        from serenityflow.nodes.face_workflow import load_sam3_model
        (model,) = load_sam3_model("sam3-base")
        assert isinstance(model, dict)
        assert model["_type"] == "sam3"

    def test_florence2_loader(self):
        from serenityflow.nodes.face_workflow import florence2_model_loader
        (model,) = florence2_model_loader("florence2-base")
        assert isinstance(model, dict)
        assert model["_type"] == "florence2"

    def test_ultralytics_loader(self):
        from serenityflow.nodes.face_workflow import ultralytics_detector_loader
        (model,) = ultralytics_detector_loader("yolov8n-face")
        assert isinstance(model, dict)
        assert model["_type"] == "ultralytics"

    def test_face_restore_loader(self):
        from serenityflow.nodes.face_workflow import face_restore_model_loader
        (model,) = face_restore_model_loader("gfpgan-1.4")
        assert isinstance(model, dict)
        assert model["_type"] == "face_restore"

    def test_joycaption_load_alias(self):
        from serenityflow.nodes.face_workflow import joycaption_load
        (model,) = joycaption_load("joycaption-alpha")
        assert isinstance(model, dict)
        assert model["_type"] == "joycaption"
