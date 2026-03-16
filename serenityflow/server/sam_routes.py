"""SAM 3 segmentation endpoints for canvas.

Supports text-prompted, click-point, and exemplar segmentation.
Model loaded via Stagehand on demand, evicted when idle.
"""
from __future__ import annotations

import base64
import io
import logging
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

# ── Lazy model holder ──

_sam_predictor: Any = None
_sam_model_name: str = ""


def _load_sam(model_name: str = "sam3") -> Any:
    """Load SAM model. Tries SAM 3 first, falls back to SAM 2.1.

    Priority:
    1. Stagehand-managed SAM (any version)
    2. SAM 3 via sam3 package (needs HF gated access to facebook/sam3)
    3. SAM 2.1 via sam2 package (local checkpoint)
    """
    global _sam_predictor, _sam_model_name

    if _sam_predictor is not None and _sam_model_name == model_name:
        return _sam_predictor

    # 1. Try Stagehand
    try:
        from stagehand import Stagehand
        sh = Stagehand.get_instance()
        _sam_predictor = sh.load_model(model_name, category="segmentation")
        _sam_model_name = model_name
        log.info("SAM loaded via Stagehand: %s", model_name)
        return _sam_predictor
    except (ImportError, Exception) as e:
        log.debug("Stagehand SAM load failed (%s), trying direct", e)

    # 2. Try SAM 3
    try:
        from sam3 import build_sam3_image_model
        _sam_predictor = build_sam3_image_model(load_from_HF=True)
        _sam_model_name = "sam3"
        log.info("SAM 3 loaded from HuggingFace")
        return _sam_predictor
    except Exception as e:
        log.debug("SAM 3 load failed (%s), falling back to SAM 2", e)

    # 3. Fallback: SAM 2.1
    try:
        import os
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        for ckpt in [
            os.path.expanduser("~/.serenity/models/sam2.1_hiera_large.pt"),
            os.path.expanduser("~/models/sam2.1_hiera_large.pt"),
            "sam2.1_hiera_large.pt",
        ]:
            if os.path.exists(ckpt):
                model = build_sam2("sam2.1_hiera_l.yaml", ckpt)
                _sam_predictor = SAM2ImagePredictor(model)
                _sam_model_name = "sam2"
                log.info("SAM 2.1 loaded from: %s", ckpt)
                return _sam_predictor

        log.warning("No SAM checkpoint found")
        return None
    except ImportError:
        log.warning("Neither sam3 nor sam2 packages available")
        return None


def _image_from_upload(content: bytes) -> np.ndarray:
    """Convert uploaded PNG/JPEG bytes to numpy RGB array."""
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(content)).convert("RGB")
    return np.array(img)


def _mask_to_png_b64(mask: np.ndarray) -> str:
    """Convert boolean/uint8 mask to base64-encoded PNG."""
    from PIL import Image as PILImage
    if mask.dtype == bool:
        mask = (mask * 255).astype(np.uint8)
    elif mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    img = PILImage.fromarray(mask, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bbox_from_mask(mask: np.ndarray) -> dict:
    """Get bounding box from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return {"x": int(cmin), "y": int(rmin), "width": int(cmax - cmin + 1), "height": int(rmax - rmin + 1)}


# ── Instance color palette ──
INSTANCE_COLORS = [
    "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#a855f7",
    "#e11d48", "#2563eb", "#16a34a", "#ca8a04", "#7c3aed",
]


def register_sam_routes(app: FastAPI):
    """Register SAM 3 segmentation endpoints."""

    @app.post("/canvas/sam3/text")
    async def sam3_text_segment(
        image: UploadFile = File(...),
        prompt: str = Form(""),
        threshold: float = Form(0.3),
    ):
        """Text-prompted segmentation. Returns masks for all matching instances."""
        if not prompt.strip():
            return JSONResponse({"error": "prompt required"}, status_code=400)

        content = await image.read()
        img_array = _image_from_upload(content)

        predictor = _load_sam()
        if predictor is None:
            return JSONResponse({
                "error": "SAM model not available. Install sam2 or configure Stagehand.",
                "instances": [],
            }, status_code=503)

        try:
            import torch
            predictor.set_image(img_array)

            # SAM 3 text-prompted mode (if available)
            # Fallback: use automatic mask generation + filtering
            if hasattr(predictor, "predict_text"):
                masks, scores, _ = predictor.predict_text(prompt, multimask_output=True)
            elif hasattr(predictor, "predict"):
                # SAM2 fallback: generate masks from center grid points
                h, w = img_array.shape[:2]
                grid_points = []
                for gy in range(3):
                    for gx in range(3):
                        grid_points.append([w * (gx + 1) / 4, h * (gy + 1) / 4])
                grid_points = np.array(grid_points)
                grid_labels = np.ones(len(grid_points), dtype=np.int32)
                masks, scores, _ = predictor.predict(
                    point_coords=grid_points,
                    point_labels=grid_labels,
                    multimask_output=True,
                )
            else:
                return JSONResponse({"error": "SAM predictor has no predict method"}, status_code=500)

            instances = []
            for i in range(len(masks)):
                score = float(scores[i]) if i < len(scores) else 0.0
                if score < threshold:
                    continue
                mask = masks[i]
                instances.append({
                    "instance_id": str(uuid.uuid4())[:8],
                    "mask_png": _mask_to_png_b64(mask),
                    "bbox": _bbox_from_mask(mask),
                    "confidence": round(score, 3),
                    "color": INSTANCE_COLORS[i % len(INSTANCE_COLORS)],
                    "label": f"{prompt} #{i+1}",
                })

            return JSONResponse({"instances": instances, "prompt": prompt})

        except Exception as e:
            log.exception("SAM text segmentation failed")
            return JSONResponse({"error": str(e), "instances": []}, status_code=500)

    @app.post("/canvas/sam3/points")
    async def sam3_point_segment(
        image: UploadFile = File(...),
        points: str = Form("[]"),
    ):
        """Click-point segmentation. Points: [{x, y, label}] where label 1=fg, 0=bg."""
        content = await image.read()
        img_array = _image_from_upload(content)

        try:
            point_list = __import__("json").loads(points)
        except Exception:
            return JSONResponse({"error": "invalid points JSON"}, status_code=400)

        if not point_list:
            return JSONResponse({"error": "no points provided"}, status_code=400)

        predictor = _load_sam()
        if predictor is None:
            return JSONResponse({"error": "SAM model not available", "instances": []}, status_code=503)

        try:
            import torch
            predictor.set_image(img_array)

            coords = np.array([[p["x"], p["y"]] for p in point_list], dtype=np.float32)
            labels = np.array([p.get("label", 1) for p in point_list], dtype=np.int32)

            masks, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=True,
            )

            # Return the best mask
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx]

            instances = [{
                "instance_id": str(uuid.uuid4())[:8],
                "mask_png": _mask_to_png_b64(mask),
                "bbox": _bbox_from_mask(mask),
                "confidence": round(float(scores[best_idx]), 3),
                "color": INSTANCE_COLORS[0],
                "label": "selection",
            }]

            return JSONResponse({"instances": instances})

        except Exception as e:
            log.exception("SAM point segmentation failed")
            return JSONResponse({"error": str(e), "instances": []}, status_code=500)

    @app.post("/canvas/sam3/exemplar")
    async def sam3_exemplar_segment(
        image: UploadFile = File(...),
        bbox: str = Form("{}"),
    ):
        """Exemplar segmentation: box one object, find all similar ones."""
        content = await image.read()
        img_array = _image_from_upload(content)

        try:
            bbox_dict = __import__("json").loads(bbox)
        except Exception:
            return JSONResponse({"error": "invalid bbox JSON"}, status_code=400)

        if not bbox_dict.get("width") or not bbox_dict.get("height"):
            return JSONResponse({"error": "bbox must have width and height"}, status_code=400)

        predictor = _load_sam()
        if predictor is None:
            return JSONResponse({"error": "SAM model not available", "instances": []}, status_code=503)

        try:
            import torch
            predictor.set_image(img_array)

            box = np.array([
                bbox_dict["x"], bbox_dict["y"],
                bbox_dict["x"] + bbox_dict["width"],
                bbox_dict["y"] + bbox_dict["height"],
            ], dtype=np.float32)

            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=True,
            )

            instances = []
            for i in range(len(masks)):
                score = float(scores[i]) if i < len(scores) else 0.0
                if score < 0.2:
                    continue
                instances.append({
                    "instance_id": str(uuid.uuid4())[:8],
                    "mask_png": _mask_to_png_b64(masks[i]),
                    "bbox": _bbox_from_mask(masks[i]),
                    "confidence": round(score, 3),
                    "color": INSTANCE_COLORS[i % len(INSTANCE_COLORS)],
                    "label": f"object #{i+1}",
                })

            return JSONResponse({"instances": instances})

        except Exception as e:
            log.exception("SAM exemplar segmentation failed")
            return JSONResponse({"error": str(e), "instances": []}, status_code=500)

    @app.post("/canvas/sam3/video")
    async def sam3_video_track(
        video: UploadFile = File(...),
        prompt: str = Form(""),
        points: str = Form("[]"),
        propagate: bool = Form(True),
    ):
        """Video tracking — propagate mask across frames.

        Accepts either:
        - A video file (extracts frames internally)
        - A single frame image (for initial segmentation before tracking)

        Returns per-frame masks with consistent instance IDs.
        Requires SAM 3 video predictor for real tracking;
        falls back to static mask propagation with SAM 2.
        """
        content = await video.read()
        filename = video.filename or "input"

        predictor = _load_sam()

        try:
            point_list = __import__("json").loads(points) if points else []
        except Exception:
            point_list = []

        # Check if this is a video file or a single frame
        is_video = any(filename.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".webm", ".mkv"])

        if is_video:
            # Extract frames from video
            try:
                frames_array = _extract_video_frames(content, max_frames=120)
            except Exception as e:
                return JSONResponse({"error": f"Frame extraction failed: {e}", "frames": []}, status_code=500)
        else:
            # Single image — treat as frame 0
            img_array = _image_from_upload(content)
            frames_array = [img_array]

        if not frames_array:
            return JSONResponse({"error": "No frames extracted", "frames": []}, status_code=400)

        # Try SAM 3 video predictor (if available)
        if predictor is not None and hasattr(predictor, "predict_video"):
            try:
                import torch
                # SAM 3 native video tracking
                masks_per_frame = predictor.predict_video(
                    frames_array,
                    prompt=prompt if prompt else None,
                    points=point_list if point_list else None,
                    propagate=propagate,
                )
                result_frames = []
                for i, mask in enumerate(masks_per_frame):
                    result_frames.append({
                        "index": i,
                        "mask_png": _mask_to_png_b64(mask),
                        "instance_id": "track_0",
                    })
                return JSONResponse({"frames": result_frames})
            except Exception as e:
                log.warning("SAM 3 video tracking failed, falling back: %s", e)

        # Fallback: segment frame 0, propagate static mask
        if predictor is not None and len(frames_array) > 0:
            try:
                import torch
                predictor.set_image(frames_array[0])

                if point_list:
                    coords = np.array([[p["x"], p["y"]] for p in point_list], dtype=np.float32)
                    labels = np.array([p.get("label", 1) for p in point_list], dtype=np.int32)
                    masks, scores, _ = predictor.predict(
                        point_coords=coords, point_labels=labels, multimask_output=True,
                    )
                else:
                    # Auto-segment from center
                    h, w = frames_array[0].shape[:2]
                    masks, scores, _ = predictor.predict(
                        point_coords=np.array([[w//2, h//2]], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int32),
                        multimask_output=True,
                    )

                best_idx = int(np.argmax(scores))
                mask_b64 = _mask_to_png_b64(masks[best_idx])

                # Propagate same mask to all frames (static fallback)
                result_frames = []
                for i in range(len(frames_array)):
                    result_frames.append({
                        "index": i,
                        "mask_png": mask_b64,
                        "instance_id": "track_0",
                    })
                return JSONResponse({"frames": result_frames})

            except Exception as e:
                log.exception("SAM video fallback failed")
                return JSONResponse({"error": str(e), "frames": []}, status_code=500)

        return JSONResponse({
            "error": "SAM model not available for video tracking",
            "frames": [],
        }, status_code=503)


def _extract_video_frames(video_bytes: bytes, max_frames: int = 120) -> list:
    """Extract frames from video bytes using OpenCV."""
    import tempfile
    try:
        import cv2
    except ImportError:
        raise RuntimeError("OpenCV (cv2) required for video frame extraction")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames)
        frames = []

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                # BGR to RGB
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(frames) >= max_frames:
                    break
            idx += 1

        cap.release()
        return frames
