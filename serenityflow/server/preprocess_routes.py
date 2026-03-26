"""Image preprocessing endpoints for ControlNet/T2I-Adapter layers.

Supports canny, depth, lineart, pose, soft_edge, tile, normal, color, scribble.
Models loaded via Stagehand on demand, evicted when idle.
"""
from __future__ import annotations

import base64
import io
import logging

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

log = logging.getLogger(__name__)

# ── Preprocessor Registry ──

PREPROCESSORS = {
    "canny":     {"name": "Canny Edge",    "needs_model": False},
    "depth":     {"name": "Depth",         "needs_model": True, "model": "depth_anything_v2"},
    "lineart":   {"name": "Lineart",       "needs_model": True, "model": "lineart"},
    "pose":      {"name": "Pose",          "needs_model": True, "model": "openpose"},
    "soft_edge": {"name": "Soft Edge",     "needs_model": True, "model": "hed"},
    "tile":      {"name": "Tile",          "needs_model": False},
    "normal":    {"name": "Normal Map",    "needs_model": True, "model": "normal_bae"},
    "color":     {"name": "Color Map",     "needs_model": False},
    "scribble":  {"name": "Scribble",      "needs_model": False},
}


def _image_from_upload(content: bytes) -> np.ndarray:
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(content)).convert("RGB")
    return np.array(img)


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image as PILImage
    if arr.ndim == 2:
        img = PILImage.fromarray(arr, mode="L")
    else:
        img = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Preprocessor Implementations ──

def _canny(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Canny edge detection. No model needed."""
    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        return edges
    except ImportError:
        raise RuntimeError("OpenCV (cv2) required for Canny edge detection")


def _depth(image: np.ndarray) -> np.ndarray:
    """Depth estimation via DepthAnything v2 or Stagehand."""
    try:
        from serenityflow.memory.stagehand import Stagehand
        sh = Stagehand.get_instance()
        model = sh.load_model("depth_anything_v2", category="preprocessor")
        result = model(image)
        return result
    except (ImportError, Exception):
        pass

    # Fallback: simple luminance-based pseudo-depth
    log.warning("No depth model available, using luminance fallback")
    gray = np.mean(image, axis=2).astype(np.uint8)
    return 255 - gray  # Invert: darker = closer


def _lineart(image: np.ndarray) -> np.ndarray:
    """Lineart extraction."""
    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Adaptive threshold for line art effect
        lineart = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return lineart
    except ImportError:
        raise RuntimeError("OpenCV required for lineart extraction")


def _pose(image: np.ndarray) -> np.ndarray:
    """OpenPose estimation."""
    try:
        from serenityflow.memory.stagehand import Stagehand
        sh = Stagehand.get_instance()
        model = sh.load_model("openpose", category="preprocessor")
        return model(image)
    except (ImportError, Exception):
        log.warning("OpenPose not available, returning blank pose map")
        return np.zeros(image.shape[:2], dtype=np.uint8)


def _soft_edge(image: np.ndarray) -> np.ndarray:
    """Soft edge detection (HED/PiDiNet style)."""
    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Gaussian blur + edge detection for soft edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        # Dilate to soften
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        soft = cv2.dilate(edges, kernel, iterations=1)
        soft = cv2.GaussianBlur(soft, (3, 3), 0)
        return soft
    except ImportError:
        raise RuntimeError("OpenCV required for soft edge detection")


def _tile(image: np.ndarray, blur_radius: int = 64) -> np.ndarray:
    """Tile preprocessor — heavy blur for tile ControlNet."""
    try:
        import cv2
        # Make blur_radius odd
        k = blur_radius | 1
        blurred = cv2.GaussianBlur(image, (k, k), 0)
        return blurred
    except ImportError:
        raise RuntimeError("OpenCV required for tile preprocessing")


def _normal(image: np.ndarray) -> np.ndarray:
    """Surface normal estimation."""
    try:
        from serenityflow.memory.stagehand import Stagehand
        sh = Stagehand.get_instance()
        model = sh.load_model("normal_bae", category="preprocessor")
        return model(image)
    except (ImportError, Exception):
        pass

    # Fallback: compute normals from grayscale gradient
    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Sobel gradients
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        # Normalize and convert to normal map (x=R, y=G, z=B)
        dz = np.ones_like(gray) * 255
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        norm[norm == 0] = 1
        normal_map = np.stack([
            ((dx / norm) * 127 + 128).astype(np.uint8),
            ((dy / norm) * 127 + 128).astype(np.uint8),
            ((dz / norm) * 127 + 128).astype(np.uint8),
        ], axis=-1)
        return normal_map
    except ImportError:
        raise RuntimeError("OpenCV required for normal map estimation")


def _color(image: np.ndarray, palette_size: int = 8) -> np.ndarray:
    """Color palette extraction — quantize to N colors."""
    try:
        from PIL import Image as PILImage
        img = PILImage.fromarray(image)
        quantized = img.quantize(colors=palette_size, method=PILImage.Quantize.MEDIANCUT)
        return np.array(quantized.convert("RGB"))
    except Exception:
        return image


def _scribble(image: np.ndarray) -> np.ndarray:
    """Convert to scribble/sketch style."""
    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        # Dodge blend
        with np.errstate(divide='ignore', invalid='ignore'):
            sketch = np.where(blurred != 0, np.clip(gray * 256 / (255 - blurred), 0, 255), 255)
        return sketch.astype(np.uint8)
    except ImportError:
        raise RuntimeError("OpenCV required for scribble conversion")


_DISPATCH = {
    "canny": _canny,
    "depth": _depth,
    "lineart": _lineart,
    "pose": _pose,
    "soft_edge": _soft_edge,
    "tile": _tile,
    "normal": _normal,
    "color": _color,
    "scribble": _scribble,
}


def register_preprocess_routes(app: FastAPI):
    """Register image preprocessing endpoints."""

    @app.post("/canvas/preprocess/{method}")
    async def preprocess(
        method: str,
        image: UploadFile = File(...),
        params: str = Form("{}"),
    ):
        """Run a preprocessor on an image. Returns processed PNG."""
        if method not in _DISPATCH:
            return Response(
                content=f'{{"error": "Unknown preprocessor: {method}"}}',
                status_code=400,
                media_type="application/json",
            )

        content = await image.read()
        img_array = _image_from_upload(content)

        try:
            import json
            param_dict = json.loads(params) if params else {}
        except Exception:
            param_dict = {}

        try:
            fn = _DISPATCH[method]
            # Pass extra params to preprocessors that accept them
            import inspect
            sig = inspect.signature(fn)
            kwargs = {}
            for key, val in param_dict.items():
                if key in sig.parameters:
                    kwargs[key] = val
            result = fn(img_array, **kwargs)
        except Exception as e:
            log.exception("Preprocessor %s failed", method)
            return Response(
                content=f'{{"error": "{str(e)}"}}',
                status_code=500,
                media_type="application/json",
            )

        png_bytes = _array_to_png_bytes(result)
        return Response(content=png_bytes, media_type="image/png")

    @app.get("/canvas/preprocessors")
    async def list_preprocessors():
        """List available preprocessors with their metadata."""
        return {name: info for name, info in PREPROCESSORS.items()}
