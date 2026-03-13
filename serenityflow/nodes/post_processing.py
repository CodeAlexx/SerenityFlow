"""Post-processing nodes -- blend, brightness, contrast, hue/saturation, etc."""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry


@registry.register(
    "ImageBlend",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image1": ("IMAGE",),
        "image2": ("IMAGE",),
        "blend_factor": ("FLOAT",),
        "blend_mode": ("STRING",),
    }},
)
def image_blend(image1, image2, blend_factor=0.5, blend_mode="normal"):
    if blend_mode == "normal":
        blended = image2
    elif blend_mode == "multiply":
        blended = image1 * image2
    elif blend_mode == "screen":
        blended = 1.0 - (1.0 - image1) * (1.0 - image2)
    elif blend_mode == "overlay":
        low = 2.0 * image1 * image2
        high = 1.0 - 2.0 * (1.0 - image1) * (1.0 - image2)
        blended = torch.where(image1 < 0.5, low, high)
    elif blend_mode == "soft_light":
        blended = (1.0 - 2.0 * image2) * image1 ** 2 + 2.0 * image2 * image1
    else:
        blended = image2
    result = image1 * (1.0 - blend_factor) + blended * blend_factor
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "ImageBrightness",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "brightness": ("FLOAT",),
    }},
)
def image_brightness(image, brightness=1.0):
    return (torch.clamp(image * brightness, 0, 1),)


@registry.register(
    "ImageContrast",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "contrast": ("FLOAT",),
    }},
)
def image_contrast(image, contrast=1.0):
    mean = image.mean(dim=(1, 2, 3), keepdim=True)
    result = (image - mean) * contrast + mean
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "ImageHueSaturation",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "hue_shift": ("FLOAT",),
        "saturation": ("FLOAT",),
        "value": ("FLOAT",),
    }},
)
def image_hue_saturation(image, hue_shift=0.0, saturation=1.0, value=1.0):
    # Convert RGB to HSV, adjust, convert back
    # image is BHWC float32 [0,1]
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    cmax = torch.max(torch.max(r, g), b)
    cmin = torch.min(torch.min(r, g), b)
    delta = cmax - cmin

    # Hue
    hue = torch.zeros_like(r)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    hue[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    hue[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    hue = hue / 6.0

    # Saturation
    sat = torch.where(cmax > 0, delta / cmax, torch.zeros_like(delta))

    # Apply adjustments
    hue = (hue + hue_shift) % 1.0
    sat = torch.clamp(sat * saturation, 0, 1)
    val = torch.clamp(cmax * value, 0, 1)

    # HSV to RGB
    h6 = hue * 6.0
    hi = h6.long() % 6
    f = h6 - h6.floor()
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)

    out_r = torch.where(hi == 0, val, torch.where(hi == 1, q, torch.where(
        hi == 2, p, torch.where(hi == 3, p, torch.where(hi == 4, t, val)))))
    out_g = torch.where(hi == 0, t, torch.where(hi == 1, val, torch.where(
        hi == 2, val, torch.where(hi == 3, q, torch.where(hi == 4, p, p)))))
    out_b = torch.where(hi == 0, p, torch.where(hi == 1, p, torch.where(
        hi == 2, t, torch.where(hi == 3, val, torch.where(hi == 4, val, q)))))

    result = torch.stack([out_r, out_g, out_b], dim=-1)
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "ImageLevels",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "black_point": ("FLOAT",),
        "white_point": ("FLOAT",),
        "gamma": ("FLOAT",),
    }},
)
def image_levels(image, black_point=0.0, white_point=1.0, gamma=1.0):
    # Remap range [black_point, white_point] -> [0, 1], apply gamma
    span = max(white_point - black_point, 1e-6)
    result = (image - black_point) / span
    result = torch.clamp(result, 0, 1)
    if gamma != 1.0:
        result = result ** (1.0 / gamma)
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "ImageGamma",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "gamma": ("FLOAT",),
    }},
)
def image_gamma(image, gamma=1.0):
    if gamma <= 0:
        gamma = 1e-6
    result = image.clamp(0, 1) ** gamma
    return (result,)


@registry.register(
    "ImageColorBalance",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "red": ("FLOAT",),
        "green": ("FLOAT",),
        "blue": ("FLOAT",),
    }},
)
def image_color_balance(image, red=1.0, green=1.0, blue=1.0):
    result = image.clone()
    result[..., 0] = result[..., 0] * red
    result[..., 1] = result[..., 1] * green
    if image.shape[-1] >= 3:
        result[..., 2] = result[..., 2] * blue
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "Canny",
    return_types=("IMAGE",),
    category="image/preprocessors",
    input_types={"required": {
        "image": ("IMAGE",),
        "low_threshold": ("FLOAT",),
        "high_threshold": ("FLOAT",),
    }},
)
def canny(image, low_threshold=0.1, high_threshold=0.3):
    # image is BHWC float32 [0,1]
    # Convert to grayscale
    gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    # gray is (B, H, W)
    gray = gray.unsqueeze(1)  # (B, 1, H, W) for conv2d

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=image.dtype, device=image.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=image.dtype, device=image.device).reshape(1, 1, 3, 3)

    gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2)
    magnitude = magnitude.squeeze(1)  # (B, H, W)

    # Threshold
    edges = torch.zeros_like(magnitude)
    edges[magnitude >= high_threshold] = 1.0
    weak = (magnitude >= low_threshold) & (magnitude < high_threshold)
    edges[weak] = 0.5

    # Promote weak edges adjacent to strong edges (simple single-pass hysteresis)
    padded = torch.nn.functional.pad(edges.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    padded = padded.squeeze(1)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[:, 1 + dy:1 + dy + edges.shape[1],
                              1 + dx:1 + dx + edges.shape[2]]
            edges[(edges == 0.5) & (neighbor == 1.0)] = 1.0
    edges[edges < 1.0] = 0.0

    # Return as 3-channel BHWC
    edges_3ch = edges.unsqueeze(-1).expand(-1, -1, -1, 3)
    return (edges_3ch,)
