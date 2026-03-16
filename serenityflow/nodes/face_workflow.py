"""Face workflow nodes -- SAM3 segmentation, Florence2 vision, face detection,
face restoration, inpaint crop/stitch, and FaceDetailer composite node."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# Helpers (internal)
# ---------------------------------------------------------------------------

_EPS = 1e-7


def _mask_bbox(mask_2d: torch.Tensor):
    """Find bounding box of nonzero region in a 2D mask. Returns (y0, x0, y1, x1) or None."""
    nonzero = mask_2d.nonzero(as_tuple=False)
    if nonzero.shape[0] == 0:
        return None
    y0 = nonzero[:, 0].min().item()
    y1 = nonzero[:, 0].max().item()
    x0 = nonzero[:, 1].min().item()
    x1 = nonzero[:, 1].max().item()
    return (y0, x0, y1, x1)


def _cosine_feather_border(h: int, w: int, border: int,
                           device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a feather mask with cosine falloff at edges. Shape (H, W), values 0-1."""
    if border <= 0:
        return torch.ones(h, w, device=device, dtype=dtype)
    mask = torch.ones(h, w, device=device, dtype=dtype)
    for i in range(min(border, h)):
        alpha = 0.5 * (1.0 - math.cos(math.pi * i / border))
        mask[i, :] *= alpha
        mask[h - 1 - i, :] *= alpha
    for i in range(min(border, w)):
        alpha = 0.5 * (1.0 - math.cos(math.pi * i / border))
        mask[:, i] *= alpha
        mask[:, w - 1 - i] *= alpha
    return mask


# ---------------------------------------------------------------------------
# SAM3 Segmentation
# ---------------------------------------------------------------------------

@registry.register(
    "LoadSAM3Model",
    return_types=("SAM3_MODEL",),
    category="face/segmentation",
    input_types={"required": {"model_name": ("STRING",)}},
)
def load_sam3_model(model_name):
    """Load a SAM3 segmentation model by name.

    Returns a handle dict for lazy loading; the bridge layer handles actual
    weight loading at execution time.
    """
    try:
        from serenityflow.bridge.serenity_api import load_sam3
        model = load_sam3(model_name)
    except (ImportError, AttributeError):
        model = {"_type": "sam3", "model_name": model_name}
    return (model,)


@registry.register(
    "SAM3Grounding",
    return_types=("MASK", "BBOXES"),
    return_names=("MASK", "BBOXES"),
    category="face/segmentation",
    input_types={
        "required": {
            "sam3_model": ("SAM3_MODEL",),
            "image": ("IMAGE",),
            "text_prompt": ("STRING",),
        },
        "optional": {
            "threshold": ("FLOAT",),
            "max_detections": ("INT",),
        },
    },
)
def sam3_grounding(sam3_model, image, text_prompt, threshold=0.3, max_detections=5):
    """Text-grounded segmentation via SAM3.

    If the model has a .segment() method, calls it directly.
    If handle/dict, delegates to bridge. Fallback: empty mask + empty bboxes.
    """
    b, h, w, c = image.shape

    # Real model with .segment()
    if hasattr(sam3_model, "segment") and callable(sam3_model.segment):
        result = sam3_model.segment(
            image, text_prompt,
            threshold=threshold, max_detections=max_detections,
        )
        if isinstance(result, tuple) and len(result) == 2:
            return result
        # If it returns just a mask, wrap it
        return (result, [])

    # Bridge handle
    if isinstance(sam3_model, dict) and sam3_model.get("_type") == "sam3":
        try:
            from serenityflow.bridge.serenity_api import run_sam3
            return run_sam3(
                sam3_model, image, text_prompt,
                threshold=threshold, max_detections=max_detections,
            )
        except (ImportError, AttributeError):
            pass

    # Fallback: empty mask and empty bbox list
    empty_mask = torch.zeros(b, h, w, dtype=torch.float32, device=image.device)
    return (empty_mask, [])


# ---------------------------------------------------------------------------
# Florence2
# ---------------------------------------------------------------------------

@registry.register(
    "Florence2ModelLoader",
    return_types=("FLORENCE2_MODEL",),
    category="face/vision",
    input_types={"required": {"model_name": ("STRING",)}},
)
def florence2_model_loader(model_name):
    """Load a Florence2 multi-task vision model by name."""
    try:
        from serenityflow.bridge.serenity_api import load_florence2
        model = load_florence2(model_name)
    except (ImportError, AttributeError):
        model = {"_type": "florence2", "model_name": model_name}
    return (model,)


@registry.register(
    "Florence2Run",
    return_types=("STRING", "MASK", "BBOXES"),
    return_names=("text", "MASK", "BBOXES"),
    category="face/vision",
    input_types={
        "required": {
            "model": ("FLORENCE2_MODEL",),
            "image": ("IMAGE",),
            "task": ("STRING",),
        },
    },
)
def florence2_run(model, image, task="caption"):
    """Run Florence2 on an image with the given task.

    Supported tasks: caption, detailed_caption, object_detection, segmentation, OCR.
    Delegates to bridge. Fallback: empty string + empty mask + empty bboxes.
    """
    b, h, w, c = image.shape

    # Real model
    if hasattr(model, "run") and callable(model.run):
        result = model.run(image, task=task)
        if isinstance(result, tuple) and len(result) == 3:
            return result

    # Bridge handle
    if isinstance(model, dict) and model.get("_type") == "florence2":
        try:
            from serenityflow.bridge.serenity_api import run_florence2
            return run_florence2(model, image, task=task)
        except (ImportError, AttributeError):
            pass

    # Fallback
    empty_mask = torch.zeros(b, h, w, dtype=torch.float32, device=image.device)
    return ("", empty_mask, [])


@registry.register(
    "Florence2Caption",
    return_types=("STRING",),
    category="face/vision",
    input_types={
        "required": {
            "model": ("FLORENCE2_MODEL",),
            "image": ("IMAGE",),
        },
        "optional": {
            "detail_level": ("STRING",),
        },
    },
)
def florence2_caption(model, image, detail_level="brief"):
    """Convenience wrapper around Florence2Run for captioning."""
    task = "detailed_caption" if detail_level == "detailed" else "caption"
    text, _, _ = florence2_run(model, image, task=task)
    return (text,)


# ---------------------------------------------------------------------------
# Face Detection (Ultralytics YOLO)
# ---------------------------------------------------------------------------

@registry.register(
    "UltralyticsDetectorLoader",
    return_types=("DETECTOR_MODEL",),
    category="face/detection",
    input_types={"required": {"model_name": ("STRING",)}},
)
def ultralytics_detector_loader(model_name):
    """Load a YOLO detector model (e.g. face detection) by name."""
    try:
        from serenityflow.bridge.serenity_api import load_ultralytics
        model = load_ultralytics(model_name)
    except (ImportError, AttributeError):
        model = {"_type": "ultralytics", "model_name": model_name}
    return (model,)


@registry.register(
    "UltralyticsDetectorRun",
    return_types=("BBOXES", "MASK"),
    return_names=("BBOXES", "MASK"),
    category="face/detection",
    input_types={
        "required": {
            "detector": ("DETECTOR_MODEL",),
            "image": ("IMAGE",),
        },
        "optional": {
            "threshold": ("FLOAT",),
        },
    },
)
def ultralytics_detector_run(detector, image, threshold=0.5):
    """Run YOLO detection. Returns bboxes and union mask of all detections."""
    b, h, w, c = image.shape

    # Real model
    if hasattr(detector, "detect") and callable(detector.detect):
        result = detector.detect(image, threshold=threshold)
        if isinstance(result, tuple) and len(result) == 2:
            return result

    # Bridge handle
    if isinstance(detector, dict) and detector.get("_type") == "ultralytics":
        try:
            from serenityflow.bridge.serenity_api import run_ultralytics
            return run_ultralytics(detector, image, threshold=threshold)
        except (ImportError, AttributeError):
            pass

    # Fallback
    empty_mask = torch.zeros(b, h, w, dtype=torch.float32, device=image.device)
    return ([], empty_mask)


# ---------------------------------------------------------------------------
# Face Restoration (GFPGAN / CodeFormer)
# ---------------------------------------------------------------------------

@registry.register(
    "FaceRestoreModelLoader",
    return_types=("FACE_RESTORE_MODEL",),
    category="face/restoration",
    input_types={"required": {"model_name": ("STRING",)}},
)
def face_restore_model_loader(model_name):
    """Load a face restoration model (GFPGAN or CodeFormer) by name."""
    try:
        from serenityflow.bridge.serenity_api import load_face_restore
        model = load_face_restore(model_name)
    except (ImportError, AttributeError):
        model = {"_type": "face_restore", "model_name": model_name}
    return (model,)


@registry.register(
    "FaceRestore",
    return_types=("IMAGE",),
    category="face/restoration",
    input_types={
        "required": {
            "face_restore_model": ("FACE_RESTORE_MODEL",),
            "image": ("IMAGE",),
        },
        "optional": {
            "fidelity": ("FLOAT",),
        },
    },
)
def face_restore(face_restore_model, image, fidelity=0.5):
    """Restore/enhance face regions in an image."""
    # Real model
    if hasattr(face_restore_model, "restore") and callable(face_restore_model.restore):
        result = face_restore_model.restore(image, fidelity=fidelity)
        return (result,)

    # Bridge handle
    if isinstance(face_restore_model, dict) and face_restore_model.get("_type") == "face_restore":
        try:
            from serenityflow.bridge.serenity_api import run_face_restore
            result = run_face_restore(face_restore_model, image, fidelity=fidelity)
            return (result,)
        except (ImportError, AttributeError):
            pass

    # Fallback: pass through unchanged
    return (image,)


# ---------------------------------------------------------------------------
# Inpaint Crop & Stitch (HeadSwapV1 pattern)
# ---------------------------------------------------------------------------

@registry.register(
    "InpaintCropImproved",
    return_types=("IMAGE", "MASK", "STITCH_DATA"),
    return_names=("IMAGE", "MASK", "STITCH_DATA"),
    category="face/inpaint",
    input_types={
        "required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
        },
        "optional": {
            "min_size": ("INT",),
            "rescale_factor": ("FLOAT",),
            "padding": ("INT",),
        },
    },
)
def inpaint_crop_improved(image, mask, min_size=512, rescale_factor=1.0, padding=32):
    """Smart crop around mask region for inpainting.

    Finds the mask bounding box, expands by padding, ensures min_size,
    crops both image and mask, and stores restore info in stitch_data.
    """
    b, h, w, c = image.shape
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] < b:
        m = m.expand(b, -1, -1)

    # Resize mask to image spatial dims if needed
    if m.shape[1] != h or m.shape[2] != w:
        m = F.interpolate(
            m.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False,
        ).squeeze(1)

    # Find bounding box from first mask in batch
    bbox = _mask_bbox(m[0])
    if bbox is None:
        # Empty mask: return full image
        stitch = {
            "x": 0, "y": 0, "crop_w": w, "crop_h": h,
            "orig_w": w, "orig_h": h,
            "rescale_factor": 1.0, "scaled_w": w, "scaled_h": h,
        }
        return (image.clone(), m.clone(), stitch)

    y0, x0, y1, x1 = bbox

    # Expand by padding
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(h - 1, y1 + padding)
    x1 = min(w - 1, x1 + padding)

    # Ensure min_size by expanding symmetrically
    crop_h = y1 - y0 + 1
    crop_w = x1 - x0 + 1

    if crop_h < min_size:
        deficit = min_size - crop_h
        expand_top = deficit // 2
        expand_bot = deficit - expand_top
        y0 = max(0, y0 - expand_top)
        y1 = min(h - 1, y1 + expand_bot)
        # If still too small (image smaller than min_size), clamp
        crop_h = y1 - y0 + 1

    if crop_w < min_size:
        deficit = min_size - crop_w
        expand_left = deficit // 2
        expand_right = deficit - expand_left
        x0 = max(0, x0 - expand_left)
        x1 = min(w - 1, x1 + expand_right)
        crop_w = x1 - x0 + 1

    # Crop image and mask
    cropped_img = image[:, y0:y1 + 1, x0:x1 + 1, :].contiguous()
    cropped_mask = m[:, y0:y1 + 1, x0:x1 + 1].contiguous()

    # Apply rescale if requested
    scaled_h = crop_h
    scaled_w = crop_w
    if rescale_factor != 1.0 and rescale_factor > 0:
        scaled_h = max(1, round(crop_h * rescale_factor))
        scaled_w = max(1, round(crop_w * rescale_factor))
        # Resize image
        img_bchw = cropped_img.permute(0, 3, 1, 2)
        img_bchw = F.interpolate(
            img_bchw, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False,
        )
        cropped_img = img_bchw.permute(0, 2, 3, 1)
        # Resize mask
        mask_b1hw = cropped_mask.unsqueeze(1)
        mask_b1hw = F.interpolate(
            mask_b1hw, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False,
        )
        cropped_mask = mask_b1hw.squeeze(1)

    stitch = {
        "x": x0, "y": y0,
        "crop_w": crop_w, "crop_h": crop_h,
        "orig_w": w, "orig_h": h,
        "rescale_factor": rescale_factor,
        "scaled_w": scaled_w, "scaled_h": scaled_h,
    }

    return (cropped_img, cropped_mask, stitch)


@registry.register(
    "InpaintStitch",
    return_types=("IMAGE",),
    category="face/inpaint",
    input_types={
        "required": {
            "original_image": ("IMAGE",),
            "inpainted_crop": ("IMAGE",),
            "stitch_data": ("STITCH_DATA",),
        },
    },
)
def inpaint_stitch(original_image, inpainted_crop, stitch_data):
    """Paste inpainted crop back into the original image with feathered blending.

    Resizes the crop back to the original region size if rescaled, then uses
    cosine-falloff feathering over an 8-16px border to avoid hard seams.
    """
    output = original_image.clone()
    x0 = stitch_data["x"]
    y0 = stitch_data["y"]
    crop_h = stitch_data["crop_h"]
    crop_w = stitch_data["crop_w"]

    # Resize inpainted crop back to original crop dimensions if rescaled
    crop = inpainted_crop
    _, ch, cw, _ = crop.shape
    if ch != crop_h or cw != crop_w:
        crop_bchw = crop.permute(0, 3, 1, 2)
        crop_bchw = F.interpolate(
            crop_bchw, size=(crop_h, crop_w), mode="bilinear", align_corners=False,
        )
        crop = crop_bchw.permute(0, 2, 3, 1)

    b = output.shape[0]
    oh = output.shape[1]
    ow = output.shape[2]

    # Clamp region to image bounds
    paste_h = min(crop_h, oh - y0)
    paste_w = min(crop_w, ow - x0)
    if paste_h <= 0 or paste_w <= 0:
        return (output,)

    crop_region = crop[:, :paste_h, :paste_w, :]
    if crop_region.shape[0] < b:
        crop_region = crop_region.expand(b, -1, -1, -1)

    # Build feather mask: cosine falloff at edges
    feather_px = min(16, paste_h // 4, paste_w // 4)
    feather = _cosine_feather_border(
        paste_h, paste_w, feather_px,
        device=output.device, dtype=output.dtype,
    )
    feather = feather.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1) for broadcasting

    # Blend
    orig_region = output[:, y0:y0 + paste_h, x0:x0 + paste_w, :]
    blended = orig_region * (1.0 - feather) + crop_region * feather
    output[:, y0:y0 + paste_h, x0:x0 + paste_w, :] = blended

    return (torch.clamp(output, 0, 1),)


# ---------------------------------------------------------------------------
# FaceDetailer (Impact Pack pattern)
# ---------------------------------------------------------------------------

@registry.register(
    "FaceDetailer",
    return_types=("IMAGE", "MASK"),
    return_names=("IMAGE", "MASK"),
    category="face/detailer",
    input_types={
        "required": {
            "image": ("IMAGE",),
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
        },
        "optional": {
            "guide_size": ("INT",),
            "max_size": ("INT",),
            "seed": ("INT",),
            "steps": ("INT",),
            "cfg": ("FLOAT",),
            "sampler_name": ("STRING",),
            "scheduler": ("STRING",),
            "denoise": ("FLOAT",),
            "bbox_threshold": ("FLOAT",),
        },
    },
)
def face_detailer(
    image, model, clip, vae, positive, negative,
    guide_size=384, max_size=1024, seed=0, steps=20, cfg=7.0,
    sampler_name="euler", scheduler="normal", denoise=0.5,
    bbox_threshold=0.5,
):
    """All-in-one face enhancement: detect faces, crop, denoise, stitch back.

    This is a composite node that packages the configuration for the executor.
    The actual KSampler step delegates to the bridge. At minimum, implements
    the face detection -> crop -> stitch pipeline structure.
    """
    b, h, w, c = image.shape

    # Step 1: Attempt face detection via bridge
    face_mask = torch.zeros(b, h, w, dtype=torch.float32, device=image.device)
    bboxes = []

    try:
        from serenityflow.bridge.serenity_api import detect_faces
        bboxes, face_mask = detect_faces(image, threshold=bbox_threshold)
    except (ImportError, AttributeError):
        pass

    if not bboxes:
        # No faces detected or bridge unavailable -- return image unchanged
        return (image, face_mask)

    # Step 2: For each detected face, crop -> denoise -> stitch
    result = image.clone()
    combined_mask = torch.zeros(b, h, w, dtype=torch.float32, device=image.device)

    for bbox in bboxes:
        bx, by, bw, bh_box = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        # Build a rectangular mask for this face
        face_region_mask = torch.zeros(b, h, w, dtype=torch.float32, device=image.device)
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w, bx + bw)
        y2 = min(h, by + bh_box)
        if x2 > x1 and y2 > y1:
            face_region_mask[:, y1:y2, x1:x2] = 1.0

        # Crop
        cropped_img, cropped_mask, stitch_data = inpaint_crop_improved(
            result, face_region_mask,
            min_size=guide_size, rescale_factor=1.0, padding=32,
        )

        # Denoise via bridge (KSampler equivalent)
        try:
            from serenityflow.bridge.serenity_api import ksampler_denoise
            denoised = ksampler_denoise(
                model=model, clip=clip, vae=vae,
                positive=positive, negative=negative,
                image=cropped_img, mask=cropped_mask,
                seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                denoise=denoise,
            )
        except (ImportError, AttributeError):
            # Bridge unavailable -- skip denoising, return cropped as-is
            denoised = cropped_img

        # Stitch back
        (result,) = inpaint_stitch(result, denoised, stitch_data)
        combined_mask = torch.max(combined_mask, face_region_mask)

    return (result, combined_mask)


# ---------------------------------------------------------------------------
# Captioning aliases
# ---------------------------------------------------------------------------

@registry.register(
    "JoyCaptionLoad",
    return_types=("JOYCAPTION_MODEL",),
    category="face/captioning",
    input_types={"required": {"model_name": ("STRING",)}},
)
def joycaption_load(model_name):
    """Alias for JoyCaptionModelLoader -- re-registered under a shorter name."""
    from serenityflow.nodes.captioning import joycaption_model_loader
    return joycaption_model_loader(model_name)


# ---------------------------------------------------------------------------
# Mask Utilities (additions)
# ---------------------------------------------------------------------------

@registry.register(
    "MasksCombineRegions",
    return_types=("MASK",),
    category="face/mask",
    input_types={
        "required": {
            "mask_1": ("MASK",),
            "mask_2": ("MASK",),
        },
        "optional": {
            "mask_3": ("MASK",),
            "mask_4": ("MASK",),
            "mask_5": ("MASK",),
        },
    },
)
def masks_combine_regions(mask_1, mask_2, mask_3=None, mask_4=None, mask_5=None):
    """Union (max) of multiple masks."""
    def _ensure_3d(m):
        if m.ndim == 2:
            return m.unsqueeze(0)
        return m

    result = torch.max(_ensure_3d(mask_1), _ensure_3d(mask_2))
    for extra in (mask_3, mask_4, mask_5):
        if extra is not None:
            result = torch.max(result, _ensure_3d(extra))
    return (result.clamp(0, 1),)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "load_sam3_model",
    "sam3_grounding",
    "florence2_model_loader",
    "florence2_run",
    "florence2_caption",
    "ultralytics_detector_loader",
    "ultralytics_detector_run",
    "face_restore_model_loader",
    "face_restore",
    "inpaint_crop_improved",
    "inpaint_stitch",
    "face_detailer",
    "joycaption_load",
    "masks_combine_regions",
]
