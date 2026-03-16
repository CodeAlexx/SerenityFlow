"""Extra ControlNet nodes -- stacking, preprocessors, T2I adapters."""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from serenityflow.nodes.registry import registry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-ControlNet Stack
# ---------------------------------------------------------------------------

@registry.register(
    "ControlNetStack",
    return_types=("CONTROL_NET_STACK",),
    category="conditioning/controlnet",
    input_types={
        "required": {
            "control_net_1": ("CONTROL_NET",),
            "image_1": ("IMAGE",),
            "strength_1": ("FLOAT",),
            "start_percent_1": ("FLOAT",),
            "end_percent_1": ("FLOAT",),
        },
        "optional": {
            "control_net_2": ("CONTROL_NET",),
            "image_2": ("IMAGE",),
            "strength_2": ("FLOAT",),
            "start_percent_2": ("FLOAT",),
            "end_percent_2": ("FLOAT",),
            "control_net_3": ("CONTROL_NET",),
            "image_3": ("IMAGE",),
            "strength_3": ("FLOAT",),
            "start_percent_3": ("FLOAT",),
            "end_percent_3": ("FLOAT",),
            "control_net_4": ("CONTROL_NET",),
            "image_4": ("IMAGE",),
            "strength_4": ("FLOAT",),
            "start_percent_4": ("FLOAT",),
            "end_percent_4": ("FLOAT",),
            "control_net_5": ("CONTROL_NET",),
            "image_5": ("IMAGE",),
            "strength_5": ("FLOAT",),
            "start_percent_5": ("FLOAT",),
            "end_percent_5": ("FLOAT",),
        },
    },
)
def controlnet_stack(
    control_net_1, image_1, strength_1=1.0, start_percent_1=0.0, end_percent_1=1.0,
    control_net_2=None, image_2=None, strength_2=1.0, start_percent_2=0.0, end_percent_2=1.0,
    control_net_3=None, image_3=None, strength_3=1.0, start_percent_3=0.0, end_percent_3=1.0,
    control_net_4=None, image_4=None, strength_4=1.0, start_percent_4=0.0, end_percent_4=1.0,
    control_net_5=None, image_5=None, strength_5=1.0, start_percent_5=0.0, end_percent_5=1.0,
):
    stack = []
    entries = [
        (control_net_1, image_1, strength_1, start_percent_1, end_percent_1),
        (control_net_2, image_2, strength_2, start_percent_2, end_percent_2),
        (control_net_3, image_3, strength_3, start_percent_3, end_percent_3),
        (control_net_4, image_4, strength_4, start_percent_4, end_percent_4),
        (control_net_5, image_5, strength_5, start_percent_5, end_percent_5),
    ]
    for cn, img, strength, start, end in entries:
        if cn is not None and img is not None:
            stack.append({
                "control_net": cn,
                "image": img,
                "strength": strength,
                "start_percent": start,
                "end_percent": end,
            })
    return (stack,)


@registry.register(
    "ApplyControlNetStack",
    return_types=("CONDITIONING", "CONDITIONING"),
    category="conditioning/controlnet",
    input_types={"required": {
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "control_stack": ("CONTROL_NET_STACK",),
    }},
)
def apply_controlnet_stack(positive, negative, control_stack):
    pos_out = list(positive)
    neg_out = list(negative)
    for entry in control_stack:
        hint = {
            "control_net": entry["control_net"],
            "hint_image": entry["image"],
            "strength": entry["strength"],
            "start_percent": entry["start_percent"],
            "end_percent": entry["end_percent"],
        }
        new_pos = []
        for c in pos_out:
            n = dict(c)
            prev = list(n.get("control_hints", []))
            prev.append(hint)
            n["control_hints"] = prev
            new_pos.append(n)
        pos_out = new_pos

        new_neg = []
        for c in neg_out:
            n = dict(c)
            prev = list(n.get("control_hints", []))
            prev.append(hint)
            n["control_hints"] = prev
            new_neg.append(n)
        neg_out = new_neg
    return (pos_out, neg_out)


# ---------------------------------------------------------------------------
# Preprocessors
# ---------------------------------------------------------------------------

@registry.register(
    "DepthAnythingPreprocessor",
    return_types=("IMAGE",),
    category="image/preprocessors",
    input_types={"required": {
        "image": ("IMAGE",),
        "model_name": ("STRING",),
    }},
)
def depth_anything_preprocessor(image, model_name="depth_anything_v2_vits"):
    # Attempt bridge-based depth estimation
    try:
        from serenityflow.bridge.serenity_api import run_preprocessor
        result = run_preprocessor("depth_anything", image, model_name=model_name)
        return (result,)
    except (ImportError, NotImplementedError, AttributeError):
        pass

    # Fallback: grayscale luminance approximation as depth proxy
    log.warning(
        "DepthAnythingPreprocessor: bridge unavailable, returning grayscale approximation"
    )
    gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    depth = gray.unsqueeze(-1).expand(-1, -1, -1, 3)
    return (depth,)


@registry.register(
    "LineartPreprocessor",
    return_types=("IMAGE",),
    category="image/preprocessors",
    input_types={"required": {
        "image": ("IMAGE",),
        "coarse": ("BOOLEAN",),
    }},
)
def lineart_preprocessor(image, coarse=False):
    # image is BHWC float32 [0,1]
    gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    gray = gray.unsqueeze(1)  # B1HW

    # Sobel edge detection
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=image.dtype, device=image.device,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=image.dtype, device=image.device,
    ).reshape(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2)

    # Threshold: coarse uses higher threshold for thicker lines
    threshold = 0.15 if coarse else 0.08
    lines = (magnitude > threshold).float()

    if not coarse:
        # Morphological thinning approximation: erode then re-threshold
        # Use a small max-pool to detect local maxima along gradient direction
        dilated = F.max_pool2d(magnitude, kernel_size=3, stride=1, padding=1)
        thin_mask = (magnitude >= dilated - 1e-6).float()
        lines = lines * thin_mask

    # Invert: white background, dark lines (standard lineart convention)
    lines = 1.0 - lines
    lines = lines.squeeze(1)  # BHW
    lines_3ch = lines.unsqueeze(-1).expand(-1, -1, -1, 3)
    return (torch.clamp(lines_3ch, 0, 1),)


@registry.register(
    "OpenPosePreprocessor",
    return_types=("IMAGE",),
    category="image/preprocessors",
    input_types={"required": {
        "image": ("IMAGE",),
        "detect_body": ("BOOLEAN",),
        "detect_hand": ("BOOLEAN",),
        "detect_face": ("BOOLEAN",),
    }},
)
def openpose_preprocessor(image, detect_body=True, detect_hand=False, detect_face=False):
    # Attempt bridge-based pose detection
    try:
        from serenityflow.bridge.serenity_api import run_preprocessor
        result = run_preprocessor(
            "openpose", image,
            detect_body=detect_body, detect_hand=detect_hand, detect_face=detect_face,
        )
        return (result,)
    except (ImportError, NotImplementedError, AttributeError):
        pass

    # Fallback: return black image with warning
    log.warning(
        "OpenPosePreprocessor: bridge unavailable, returning black placeholder image"
    )
    black = torch.zeros_like(image)
    return (black,)


@registry.register(
    "TilePreprocessor",
    return_types=("IMAGE",),
    category="image/preprocessors",
    input_types={"required": {
        "image": ("IMAGE",),
        "pyrUp_iters": ("INT",),
    }},
)
def tile_preprocessor(image, pyrUp_iters=3):
    # image is BHWC float32 [0,1]
    img = image.permute(0, 3, 1, 2)  # BCHW
    _, _, h, w = img.shape

    # Repeatedly downscale then upscale (pyramid blur)
    current = img
    for _ in range(pyrUp_iters):
        # Downscale by 2x
        dh = max(current.shape[2] // 2, 1)
        dw = max(current.shape[3] // 2, 1)
        current = F.interpolate(current, size=(dh, dw), mode="bilinear", align_corners=False)

    # Upscale back to original resolution
    current = F.interpolate(current, size=(h, w), mode="bilinear", align_corners=False)

    result = current.permute(0, 2, 3, 1)  # BHWC
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "SoftEdgePreprocessor",
    return_types=("IMAGE",),
    category="image/preprocessors",
    input_types={"required": {
        "image": ("IMAGE",),
        "safe": ("BOOLEAN",),
    }},
)
def soft_edge_preprocessor(image, safe=False):
    # image is BHWC float32 [0,1]
    gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    gray = gray.unsqueeze(1)  # B1HW

    # Sobel edge detection
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=image.dtype, device=image.device,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=image.dtype, device=image.device,
    ).reshape(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2)

    # Normalize to 0-1
    mag_max = magnitude.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    edges = magnitude / mag_max

    # Gaussian blur for soft edges
    blur_radius = 3 if safe else 2
    ksize = blur_radius * 2 + 1
    x_coord = torch.arange(ksize, dtype=image.dtype, device=image.device) - blur_radius
    sigma = blur_radius * 0.5
    kernel_1d = torch.exp(-0.5 * (x_coord / max(sigma, 1e-6)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.reshape(1, 1, ksize, ksize)

    pad = blur_radius
    edges = F.conv2d(
        F.pad(edges, (pad, pad, pad, pad), mode="reflect"),
        kernel_2d,
    )

    if safe:
        # Clamp to reduce noise in safe mode
        edges = torch.clamp(edges * 1.5, 0, 1)

    edges = edges.squeeze(1)  # BHW
    edges_3ch = edges.unsqueeze(-1).expand(-1, -1, -1, 3)
    return (torch.clamp(edges_3ch, 0, 1),)


# ---------------------------------------------------------------------------
# T2I Adapter
# ---------------------------------------------------------------------------

@registry.register(
    "T2IAdapterLoader",
    return_types=("T2I_ADAPTER",),
    category="loaders",
    input_types={"required": {"model_name": ("STRING",)}},
)
def t2i_adapter_loader(model_name):
    try:
        from serenityflow.bridge.serenity_api import load_t2i_adapter
        from serenityflow.bridge.model_paths import get_model_paths
        paths = get_model_paths()
        path = paths.find(model_name, "t2i_adapter")
        return (load_t2i_adapter(path),)
    except (ImportError, NotImplementedError, AttributeError):
        # Return a path handle for deferred loading
        return ({"model_name": model_name, "type": "t2i_adapter"},)


@registry.register(
    "T2IAdapterApply",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "t2i_adapter": ("T2I_ADAPTER",),
        "image": ("IMAGE",),
        "strength": ("FLOAT",),
    }},
)
def t2i_adapter_apply(conditioning, t2i_adapter, image, strength=1.0):
    out = []
    for c in conditioning:
        n = dict(c)
        prev = list(n.get("t2i_adapters", []))
        prev.append({
            "adapter": t2i_adapter,
            "image": image,
            "strength": strength,
        })
        n["t2i_adapters"] = prev
        out.append(n)
    return (out,)
