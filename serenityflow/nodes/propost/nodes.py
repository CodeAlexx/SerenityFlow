"""ProPost nodes — film grain, vignette, radial blur, depth map blur, LUT application.

Ported from https://github.com/digitaljohn/comfyui-propost (MIT License).
Film grain engine vendored from https://github.com/larspontoppidan/filmgrainer (MIT License).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from serenityflow.nodes.propost.filmgrainer import filmgrainer
from serenityflow.nodes.propost.lut_utils import read_cube
from serenityflow.nodes.registry import registry

# ---------------------------------------------------------------------------
# LUT directory — configurable via SERENITYFLOW_LUT_DIR env var
# ---------------------------------------------------------------------------
_DEFAULT_LUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "luts")
LUT_DIR = os.environ.get("SERENITYFLOW_LUT_DIR", os.path.normpath(_DEFAULT_LUT_DIR))
os.makedirs(LUT_DIR, exist_ok=True)


def _list_luts() -> list[str]:
    """Return sorted list of .cube filenames in LUT_DIR."""
    if not os.path.isdir(LUT_DIR):
        return []
    return sorted(f for f in os.listdir(LUT_DIR) if f.lower().endswith(".cube"))


# ---------------------------------------------------------------------------
# Blur utilities (ported from propost utils/processing.py)
# ---------------------------------------------------------------------------

def _generate_blurred_images(image: np.ndarray, blur_strength: float,
                             steps: int, focus_spread: float = 1.0) -> list[np.ndarray]:
    blurred = []
    for step in range(1, steps + 1):
        blur_factor = (step / steps) ** focus_spread * blur_strength
        blur_size = max(1, int(blur_factor))
        if blur_size % 2 == 0:
            blur_size += 1
        blurred.append(cv2.GaussianBlur(image, (blur_size, blur_size), 0))
    return blurred


def _apply_blurred_images(image: np.ndarray, blurred_images: list[np.ndarray],
                          mask: np.ndarray) -> np.ndarray:
    steps = len(blurred_images)
    final = np.zeros_like(image)
    step_size = 1.0 / steps
    for i, blurred in enumerate(blurred_images):
        current_mask = np.clip((mask - i * step_size) * steps, 0, 1)
        next_mask = np.clip((mask - (i + 1) * step_size) * steps, 0, 1)
        blend_mask = current_mask - next_mask
        final += blend_mask[:, :, np.newaxis] * blurred
    final += (1 - np.clip(mask * steps, 0, 1))[:, :, np.newaxis] * image
    return final


# ===========================================================================
# Node: ProPostFilmGrain
# ===========================================================================

GRAIN_TYPES = ["Fine", "Fine Simple", "Coarse", "Coarser"]


@registry.register(
    "ProPostFilmGrain",
    return_types=("IMAGE",),
    category="ProPost",
    display_name="ProPost Film Grain",
    input_types={"required": {
        "image": ("IMAGE",),
        "gray_scale": ("BOOLEAN", {"default": False}),
        "grain_type": (GRAIN_TYPES,),
        "grain_sat": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        "grain_power": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
        "shadows": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
        "highs": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
        "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        "sharpen": ("INT", {"default": 0, "min": 0, "max": 10}),
        "src_gamma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
    }},
)
def propost_film_grain(image: torch.Tensor, gray_scale: bool, grain_type: str,
                       grain_sat: float, grain_power: float, shadows: float,
                       highs: float, scale: float, sharpen: int, src_gamma: float,
                       seed: int):
    batch_size = image.shape[0]
    grain_type_index = GRAIN_TYPES.index(grain_type) + 1
    result = torch.zeros_like(image)
    for b in range(batch_size):
        arr = image[b].cpu().numpy()
        out = filmgrainer.process(
            arr, scale, src_gamma, grain_power, shadows, highs,
            grain_type_index, grain_sat, gray_scale, sharpen, seed + b,
        )
        result[b] = torch.from_numpy(out)
    return (result,)


# ===========================================================================
# Node: ProPostVignette
# ===========================================================================

@registry.register(
    "ProPostVignette",
    return_types=("IMAGE",),
    category="ProPost",
    display_name="ProPost Vignette",
    input_types={"required": {
        "image": ("IMAGE",),
        "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
    }},
)
def propost_vignette(image: torch.Tensor, intensity: float = 1.0,
                     center_x: float = 0.5, center_y: float = 0.5):
    if intensity == 0:
        return (image,)

    batch_size, height, width, _ = image.shape

    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x - (2 * center_x - 1), y - (2 * center_y - 1))

    distances = [
        np.sqrt((0 - center_x) ** 2 + (0 - center_y) ** 2),
        np.sqrt((1 - center_x) ** 2 + (0 - center_y) ** 2),
        np.sqrt((0 - center_x) ** 2 + (1 - center_y) ** 2),
        np.sqrt((1 - center_x) ** 2 + (1 - center_y) ** 2),
    ]
    max_dist = np.max(distances)

    radius = np.sqrt(X ** 2 + Y ** 2) / (max_dist * np.sqrt(2))
    opacity = np.clip(intensity, 0, 1)
    vignette = 1 - radius * opacity
    # Convert to torch for broadcasting — shape (1, H, W, 1)
    vig_t = torch.from_numpy(vignette.astype(np.float32)).unsqueeze(0).unsqueeze(-1)

    result = torch.clamp(image * vig_t, 0, 1)
    return (result,)


# ===========================================================================
# Node: ProPostRadialBlur
# ===========================================================================

@registry.register(
    "ProPostRadialBlur",
    return_types=("IMAGE",),
    category="ProPost",
    display_name="ProPost Radial Blur",
    input_types={"required": {
        "image": ("IMAGE",),
        "blur_strength": ("FLOAT", {"default": 64.0, "min": 0.0, "max": 256.0, "step": 1.0}),
        "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        "focus_spread": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
        "steps": ("INT", {"default": 5, "min": 1, "max": 32}),
    }},
)
def propost_radial_blur(image: torch.Tensor, blur_strength: float = 64.0,
                        center_x: float = 0.5, center_y: float = 0.5,
                        focus_spread: float = 1.0, steps: int = 5):
    batch_size, height, width, _ = image.shape
    result = torch.zeros_like(image)

    c_x, c_y = int(width * center_x), int(height * center_y)
    distances = [
        np.sqrt((c_x - 0) ** 2 + (c_y - 0) ** 2),
        np.sqrt((c_x - width) ** 2 + (c_y - 0) ** 2),
        np.sqrt((c_x - 0) ** 2 + (c_y - height) ** 2),
        np.sqrt((c_x - width) ** 2 + (c_y - height) ** 2),
    ]
    max_dist = max(distances)
    X, Y = np.meshgrid(np.arange(width) - c_x, np.arange(height) - c_y)
    radial_mask = np.sqrt(X ** 2 + Y ** 2) / max_dist

    for b in range(batch_size):
        arr = image[b].cpu().numpy()
        blurred_imgs = _generate_blurred_images(arr, blur_strength, steps, focus_spread)
        out = _apply_blurred_images(arr, blurred_imgs, radial_mask)
        result[b] = torch.from_numpy(np.clip(out, 0, 1).astype(np.float32))
    return (result,)


# ===========================================================================
# Node: ProPostDepthMapBlur
# ===========================================================================

@registry.register(
    "ProPostDepthMapBlur",
    return_types=("IMAGE", "MASK"),
    return_names=["image", "mask"],
    category="ProPost",
    display_name="ProPost Depth Map Blur",
    input_types={"required": {
        "image": ("IMAGE",),
        "depth_map": ("IMAGE",),
        "blur_strength": ("FLOAT", {"default": 64.0, "min": 0.0, "max": 256.0, "step": 1.0}),
        "focal_depth": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "focus_spread": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.1}),
        "steps": ("INT", {"default": 5, "min": 1, "max": 32}),
        "focal_range": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "mask_blur": ("INT", {"default": 1, "min": 1, "max": 127, "step": 2}),
    }},
)
def propost_depth_map_blur(image: torch.Tensor, depth_map: torch.Tensor,
                           blur_strength: float = 64.0, focal_depth: float = 1.0,
                           focus_spread: float = 1.0, steps: int = 5,
                           focal_range: float = 0.0, mask_blur: int = 1):
    batch_size, height, width, _ = image.shape
    image_result = torch.zeros_like(image)
    mask_result = torch.zeros((batch_size, height, width), dtype=torch.float32)

    for b in range(batch_size):
        arr = image[b].cpu().numpy()
        depth_arr = depth_map[b].cpu().numpy()

        # Normalise depth if needed
        if depth_arr.max() > 1:
            depth_arr = depth_arr.astype(np.float32) / 255

        # Resize depth to match image
        depth_resized = cv2.resize(depth_arr, (width, height), interpolation=cv2.INTER_LINEAR)
        if len(depth_resized.shape) > 2:
            depth_resized = cv2.cvtColor(depth_resized, cv2.COLOR_BGR2GRAY)

        # Build depth mask
        depth_mask = np.abs(depth_resized - focal_depth)
        max_val = np.max(depth_mask)
        if max_val > 0:
            depth_mask = np.clip(depth_mask / max_val, 0, 1)

        # Apply focal range
        depth_mask[depth_mask < focal_range] = 0
        if focal_range < 1:
            above = depth_mask >= focal_range
            depth_mask[above] = (depth_mask[above] - focal_range) / (1 - focal_range)

        # Smooth the mask
        if mask_blur > 1:
            blur_k = mask_blur if mask_blur % 2 == 1 else mask_blur + 1
            depth_mask = cv2.GaussianBlur(depth_mask, (blur_k, blur_k), 0)

        # Generate and apply blurred images
        blurred_imgs = _generate_blurred_images(arr, blur_strength, steps, focus_spread)
        final = _apply_blurred_images(arr, blurred_imgs, depth_mask)

        image_result[b] = torch.from_numpy(np.clip(final, 0, 1).astype(np.float32))
        mask_result[b] = torch.from_numpy(depth_mask.astype(np.float32))

    return (image_result, mask_result)


# ===========================================================================
# Node: ProPostApplyLUT
# ===========================================================================

@registry.register(
    "ProPostApplyLUT",
    return_types=("IMAGE",),
    category="ProPost",
    display_name="ProPost Apply LUT",
    input_types=lambda: {"required": {
        "image": ("IMAGE",),
        "lut_name": (_list_luts(),),
        "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "log": ("BOOLEAN", {"default": False}),
    }},
)
def propost_apply_lut(image: torch.Tensor, lut_name: str,
                      strength: float = 1.0, log: bool = False):
    if strength == 0:
        return (image,)

    lut_path = os.path.join(LUT_DIR, lut_name)
    lut = read_cube(lut_path, clip=True)

    batch_size = image.shape[0]
    result = torch.zeros_like(image)

    for b in range(batch_size):
        arr = image[b].cpu().numpy().copy()

        # Apply domain scaling
        is_non_default = not np.array_equal(
            lut.domain, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        )
        dom_scale = None
        if is_non_default:
            dom_scale = lut.domain[1] - lut.domain[0]
            arr = arr * dom_scale + lut.domain[0]
        if log:
            arr = arr ** (1 / 2.2)

        arr = lut.apply(arr)

        if log:
            arr = arr ** 2.2
        if is_non_default:
            arr = (arr - lut.domain[0]) / dom_scale

        blended = (1 - strength) * image[b].cpu().numpy() + strength * arr
        result[b] = torch.from_numpy(np.clip(blended, 0, 1).astype(np.float32))

    return (result,)
