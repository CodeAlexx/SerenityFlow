"""Layer compositing nodes -- Photoshop-style blend modes, color adjustments,
filters, layer style effects, and utility generators."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# Helpers (internal)
# ---------------------------------------------------------------------------

_EPS = 1e-7


def _gaussian_kernel_2d(radius: int, sigma: float, device: torch.device, dtype: torch.dtype):
    """Build a 2D Gaussian kernel of size (2*radius+1)^2."""
    ksize = radius * 2 + 1
    x = torch.arange(ksize, dtype=dtype, device=device) - radius
    kernel_1d = torch.exp(-0.5 * (x / max(sigma, _EPS)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)


def _gaussian_blur(img_bchw: torch.Tensor, radius: int, sigma: float) -> torch.Tensor:
    """Gaussian blur on BCHW tensor."""
    if radius == 0:
        return img_bchw
    kernel = _gaussian_kernel_2d(radius, sigma, img_bchw.device, img_bchw.dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(img_bchw.shape[1], 1, -1, -1)
    pad = radius
    padded = F.pad(img_bchw, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel, groups=img_bchw.shape[1])


def _blend_fn(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply blend mode between background *a* and foreground *b*. Both float32 0-1."""
    if mode == "normal":
        return b
    if mode == "multiply":
        return a * b
    if mode == "screen":
        return 1.0 - (1.0 - a) * (1.0 - b)
    if mode == "overlay":
        low = 2.0 * a * b
        high = 1.0 - 2.0 * (1.0 - a) * (1.0 - b)
        return torch.where(a < 0.5, low, high)
    if mode == "soft_light":
        return (1.0 - 2.0 * b) * a ** 2 + 2.0 * b * a
    if mode == "hard_light":
        low = 2.0 * a * b
        high = 1.0 - 2.0 * (1.0 - a) * (1.0 - b)
        return torch.where(b < 0.5, low, high)
    if mode == "color_dodge":
        return a / (1.0 - b + _EPS)
    if mode == "color_burn":
        return 1.0 - (1.0 - a) / (b + _EPS)
    if mode == "darken":
        return torch.min(a, b)
    if mode == "lighten":
        return torch.max(a, b)
    if mode == "difference":
        return torch.abs(a - b)
    if mode == "exclusion":
        return a + b - 2.0 * a * b
    if mode == "add":
        return a + b
    if mode == "subtract":
        return a - b
    # fallback: normal
    return b


def _resize_fg(fg: torch.Tensor, target_h: int, target_w: int, scale: float,
               antialias: bool) -> torch.Tensor:
    """Resize fg (BHWC) by *scale* factor using bilinear interpolation."""
    if scale == 1.0:
        return fg
    _, fh, fw, _ = fg.shape
    new_h = max(1, round(fh * scale))
    new_w = max(1, round(fw * scale))
    fg_bchw = fg.permute(0, 3, 1, 2)
    if antialias and scale < 1.0:
        fg_bchw = F.interpolate(fg_bchw, size=(new_h, new_w), mode="bilinear",
                                align_corners=False, antialias=True)
    else:
        fg_bchw = F.interpolate(fg_bchw, size=(new_h, new_w), mode="bilinear",
                                align_corners=False)
    return fg_bchw.permute(0, 2, 3, 1)


def _rotate_image(img_bhwc: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate image by angle in degrees (counter-clockwise). Returns BHWC."""
    if angle_deg % 360 == 0:
        return img_bhwc
    b, h, w, c = img_bhwc.shape
    theta_rad = math.radians(-angle_deg)  # negative for standard CCW
    cos_a = math.cos(theta_rad)
    sin_a = math.sin(theta_rad)
    # Build 2x3 affine matrix
    theta = torch.tensor([[cos_a, -sin_a, 0.0],
                          [sin_a, cos_a, 0.0]],
                         dtype=img_bhwc.dtype, device=img_bhwc.device)
    theta = theta.unsqueeze(0).expand(b, -1, -1)
    grid = F.affine_grid(theta, [b, c, h, w], align_corners=False)
    img_bchw = img_bhwc.permute(0, 3, 1, 2)
    rotated = F.grid_sample(img_bchw, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)
    return rotated.permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# ImageBlendAdvanced
# ---------------------------------------------------------------------------

@registry.register(
    "ImageBlendAdvanced",
    return_types=("IMAGE", "MASK"),
    return_names=("IMAGE", "MASK"),
    category="image/compositing",
    input_types={
        "required": {
            "background": ("IMAGE",),
            "foreground": ("IMAGE",),
            "blend_mode": ("STRING",),
            "opacity": ("FLOAT",),
        },
        "optional": {
            "mask": ("MASK",),
            "x_percent": ("FLOAT",),
            "y_percent": ("FLOAT",),
            "scale": ("FLOAT",),
            "rotation": ("FLOAT",),
            "antialias": ("BOOLEAN",),
        },
    },
)
def image_blend_advanced(
    background,
    foreground,
    blend_mode="normal",
    opacity=1.0,
    mask=None,
    x_percent=50.0,
    y_percent=50.0,
    scale=1.0,
    rotation=0.0,
    antialias=True,
):
    bg = background.clone()
    b, bh, bw, c = bg.shape
    fg = foreground

    # Scale foreground
    if scale != 1.0:
        fg = _resize_fg(fg, bh, bw, scale, antialias)

    # Rotate foreground
    if rotation % 360 != 0:
        fg = _rotate_image(fg, rotation)

    _, fh, fw, _ = fg.shape

    # Compute top-left placement from percent (center of fg placed at percent of bg)
    cx = int(bw * x_percent / 100.0)
    cy = int(bh * y_percent / 100.0)
    x0 = cx - fw // 2
    y0 = cy - fh // 2

    # Compute overlap region
    sx = max(0, -x0)
    sy = max(0, -y0)
    dx = max(0, x0)
    dy = max(0, y0)
    rw = min(fw - sx, bw - dx)
    rh = min(fh - sy, bh - dy)

    # Build composite mask (full ones then multiply with user mask)
    comp_mask = torch.zeros(b, bh, bw, dtype=bg.dtype, device=bg.device)

    if rw > 0 and rh > 0:
        fg_region = fg[:, sy:sy + rh, sx:sx + rw, :]
        # Broadcast batch if fg has fewer frames
        if fg_region.shape[0] < b:
            fg_region = fg_region.expand(b, -1, -1, -1)

        bg_region = bg[:, dy:dy + rh, dx:dx + rw, :]

        blended = _blend_fn(bg_region, fg_region, blend_mode)
        blended = torch.clamp(blended, 0, 1)

        # Build alpha for this region
        alpha = torch.ones(b, rh, rw, dtype=bg.dtype, device=bg.device) * opacity
        if mask is not None:
            m = mask
            if m.ndim == 2:
                m = m.unsqueeze(0)
            if m.shape[0] < b:
                m = m.expand(b, -1, -1)
            # Resize mask to fg size if needed, then crop to region
            if m.shape[1] != fh or m.shape[2] != fw:
                m = F.interpolate(m.unsqueeze(1), size=(fh, fw), mode="bilinear",
                                  align_corners=False).squeeze(1)
            m_region = m[:, sy:sy + rh, sx:sx + rw]
            alpha = alpha * m_region

        alpha_4d = alpha.unsqueeze(-1)  # BHW -> BHWC broadcastable
        bg[:, dy:dy + rh, dx:dx + rw, :] = (
            bg_region * (1.0 - alpha_4d) + blended * alpha_4d
        )
        comp_mask[:, dy:dy + rh, dx:dx + rw] = alpha

    result = torch.clamp(bg, 0, 1)
    return (result, comp_mask)


# ---------------------------------------------------------------------------
# Color Adjustment Additions
# ---------------------------------------------------------------------------

@registry.register(
    "Exposure",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "ev": ("FLOAT",),
    }},
)
def exposure(image, ev=0.0):
    result = image * (2.0 ** ev)
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "ColorTemperature",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "kelvin": ("FLOAT",),
    }},
)
def color_temperature(image, kelvin=6500.0):
    # Neutral at 6500K.  Warm (lower K) boosts R, reduces B.  Cool (higher K) boosts B, reduces R.
    shift = (kelvin - 6500.0) / 6500.0  # normalised offset
    result = image.clone()
    result[..., 0] = result[..., 0] * (1.0 - shift * 0.3)  # R: less at high K
    if result.shape[-1] >= 3:
        result[..., 2] = result[..., 2] * (1.0 + shift * 0.3)  # B: more at high K
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "LUTApply",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "lut_path": ("STRING",),
    }},
)
def lut_apply(image, lut_path=""):
    """Apply a .cube 3D LUT file with trilinear interpolation."""
    lut, size = _parse_cube_lut(lut_path)  # (size, size, size, 3) on CPU
    if lut is None:
        return (image,)  # Invalid path or parse error -- pass through
    lut = lut.to(device=image.device, dtype=image.dtype)
    result = _trilinear_lut(image, lut, size)
    return (torch.clamp(result, 0, 1),)


def _parse_cube_lut(path: str):
    """Parse a .cube LUT file.  Returns (tensor[S,S,S,3], size) or (None, 0)."""
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except (OSError, IOError):
        return None, 0

    size = 0
    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if line.upper().startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX")):
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    if size == 0 or len(data) != size ** 3:
        return None, 0

    lut = torch.tensor(data, dtype=torch.float32).reshape(size, size, size, 3)
    return lut, size


def _trilinear_lut(image: torch.Tensor, lut: torch.Tensor, size: int) -> torch.Tensor:
    """Apply 3D LUT with trilinear interpolation.  image: BHWC, lut: (S,S,S,3)."""
    b, h, w, c = image.shape
    rgb = image[..., :3].clamp(0, 1) * (size - 1)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b_ch = rgb[..., 2]

    r0 = r.long().clamp(0, size - 2)
    g0 = g.long().clamp(0, size - 2)
    b0 = b_ch.long().clamp(0, size - 2)
    r1 = r0 + 1
    g1 = g0 + 1
    b1 = b0 + 1

    fr = (r - r0.float()).unsqueeze(-1)
    fg = (g - g0.float()).unsqueeze(-1)
    fb = (b_ch - b0.float()).unsqueeze(-1)

    # 8 corners of the cube
    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    # Trilinear interpolation
    out = (c000 * (1 - fr) * (1 - fg) * (1 - fb) +
           c001 * (1 - fr) * (1 - fg) * fb +
           c010 * (1 - fr) * fg * (1 - fb) +
           c011 * (1 - fr) * fg * fb +
           c100 * fr * (1 - fg) * (1 - fb) +
           c101 * fr * (1 - fg) * fb +
           c110 * fr * fg * (1 - fb) +
           c111 * fr * fg * fb)

    if c > 3:
        out = torch.cat([out, image[..., 3:]], dim=-1)
    return out


@registry.register(
    "AutoAdjust",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {"image": ("IMAGE",)}},
)
def auto_adjust(image):
    """Auto levels: histogram stretch per channel."""
    result = image.clone()
    for ch in range(min(3, image.shape[-1])):
        channel = result[..., ch]
        lo = channel.min()
        hi = channel.max()
        span = hi - lo
        if span > _EPS:
            result[..., ch] = (channel - lo) / span
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "Negative",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {"image": ("IMAGE",)}},
)
def negative(image):
    return (1.0 - image,)


# ---------------------------------------------------------------------------
# Image Filters
# ---------------------------------------------------------------------------

@registry.register(
    "AddGrain",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "amount": ("FLOAT",),
        "grain_size": ("INT",),
        "seed": ("INT",),
    }},
)
def add_grain(image, amount=0.1, grain_size=1, seed=0):
    if amount <= 0:
        return (image,)
    b, h, w, c = image.shape
    gen = torch.Generator(device=image.device).manual_seed(seed)
    gh = max(1, h // max(1, grain_size))
    gw = max(1, w // max(1, grain_size))
    noise = torch.randn(b, gh, gw, c, generator=gen, device=image.device, dtype=image.dtype)
    if grain_size > 1:
        noise = noise.permute(0, 3, 1, 2)
        noise = F.interpolate(noise, size=(h, w), mode="nearest")
        noise = noise.permute(0, 2, 3, 1)
    result = image + noise * amount
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "ChannelShake",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "r_offset_x": ("INT",), "r_offset_y": ("INT",),
        "g_offset_x": ("INT",), "g_offset_y": ("INT",),
        "b_offset_x": ("INT",), "b_offset_y": ("INT",),
    }},
)
def channel_shake(image, r_offset_x=0, r_offset_y=0, g_offset_x=0, g_offset_y=0,
                  b_offset_x=0, b_offset_y=0):
    offsets = [(r_offset_x, r_offset_y), (g_offset_x, g_offset_y), (b_offset_x, b_offset_y)]
    channels = []
    for ch_idx, (ox, oy) in enumerate(offsets):
        if ch_idx >= image.shape[-1]:
            break
        ch = image[..., ch_idx]  # B, H, W
        if ox != 0 or oy != 0:
            ch = torch.roll(ch, shifts=(oy, ox), dims=(1, 2))
        channels.append(ch)
    # Handle images with >3 channels
    for ch_idx in range(3, image.shape[-1]):
        channels.append(image[..., ch_idx])
    result = torch.stack(channels, dim=-1)
    return (torch.clamp(result, 0, 1),)


# ---------------------------------------------------------------------------
# Layer Style Effects
# ---------------------------------------------------------------------------

@registry.register(
    "DropShadow",
    return_types=("IMAGE",),
    category="image/compositing",
    input_types={
        "required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "angle": ("FLOAT",),
            "distance": ("INT",),
            "blur_radius": ("INT",),
            "shadow_r": ("FLOAT",),
            "shadow_g": ("FLOAT",),
            "shadow_b": ("FLOAT",),
            "opacity": ("FLOAT",),
        },
    },
)
def drop_shadow(image, mask, angle=135.0, distance=10, blur_radius=5,
                shadow_r=0.0, shadow_g=0.0, shadow_b=0.0, opacity=0.5):
    b, h, w, c = image.shape
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] < b:
        m = m.expand(b, -1, -1)
    # Resize mask to image size if needed
    if m.shape[1] != h or m.shape[2] != w:
        m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear",
                          align_corners=False).squeeze(1)

    # Compute offset from angle + distance
    rad = math.radians(angle)
    ox = round(distance * math.cos(rad))
    oy = round(-distance * math.sin(rad))  # negative because y-axis is down

    # Shift mask
    shadow_mask = torch.roll(m, shifts=(oy, ox), dims=(1, 2))

    # Blur shadow mask
    if blur_radius > 0:
        sigma = blur_radius / 2.0
        shadow_mask = _gaussian_blur(shadow_mask.unsqueeze(1), blur_radius, sigma).squeeze(1)

    shadow_mask = shadow_mask.clamp(0, 1) * opacity

    # Build shadow color
    shadow_color = torch.tensor([shadow_r, shadow_g, shadow_b],
                                dtype=image.dtype, device=image.device)
    shadow_layer = shadow_color.view(1, 1, 1, 3).expand(b, h, w, -1)

    # Composite: shadow behind image
    alpha = shadow_mask.unsqueeze(-1)
    # Where the original mask is opaque, show image; where it isn't, show shadow
    orig_mask = m.unsqueeze(-1)
    result = image * orig_mask + shadow_layer * alpha * (1.0 - orig_mask) + image * (1.0 - orig_mask) * (1.0 - alpha)

    return (torch.clamp(result, 0, 1),)


@registry.register(
    "InnerGlow",
    return_types=("IMAGE",),
    category="image/compositing",
    input_types={
        "required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "glow_r": ("FLOAT",),
            "glow_g": ("FLOAT",),
            "glow_b": ("FLOAT",),
            "size": ("INT",),
            "opacity": ("FLOAT",),
        },
    },
)
def inner_glow(image, mask, glow_r=1.0, glow_g=1.0, glow_b=1.0, size=10, opacity=0.5):
    b, h, w, c = image.shape
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] < b:
        m = m.expand(b, -1, -1)
    if m.shape[1] != h or m.shape[2] != w:
        m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear",
                          align_corners=False).squeeze(1)

    # Erode mask to find inner edge
    if size > 0:
        inv = 1.0 - m.unsqueeze(1)
        ksize = size * 2 + 1
        kernel = torch.ones(1, 1, ksize, ksize, dtype=image.dtype, device=image.device)
        padded = F.pad(inv, (size, size, size, size), mode="constant", value=0.0)
        dilated = F.conv2d(padded, kernel, padding=0)
        eroded = 1.0 - (dilated > 0.5).float()
        eroded = eroded.squeeze(1)
        # Inner edge = original mask - eroded mask
        edge = (m - eroded).clamp(0, 1)
    else:
        edge = torch.zeros_like(m)

    # Blur edge for soft glow
    if size > 0:
        sigma = max(size / 3.0, 0.5)
        edge = _gaussian_blur(edge.unsqueeze(1), size, sigma).squeeze(1)
        edge = edge.clamp(0, 1)

    glow_color = torch.tensor([glow_r, glow_g, glow_b],
                              dtype=image.dtype, device=image.device)
    glow_layer = glow_color.view(1, 1, 1, 3).expand(b, h, w, -1)

    alpha = (edge * opacity * m).unsqueeze(-1)
    result = image * (1.0 - alpha) + glow_layer * alpha
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "InnerShadow",
    return_types=("IMAGE",),
    category="image/compositing",
    input_types={
        "required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "angle": ("FLOAT",),
            "distance": ("INT",),
            "blur_radius": ("INT",),
            "shadow_r": ("FLOAT",),
            "shadow_g": ("FLOAT",),
            "shadow_b": ("FLOAT",),
            "opacity": ("FLOAT",),
        },
    },
)
def inner_shadow(image, mask, angle=135.0, distance=5, blur_radius=5,
                 shadow_r=0.0, shadow_g=0.0, shadow_b=0.0, opacity=0.5):
    b, h, w, c = image.shape
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] < b:
        m = m.expand(b, -1, -1)
    if m.shape[1] != h or m.shape[2] != w:
        m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear",
                          align_corners=False).squeeze(1)

    # Invert mask, shift, blur, then intersect with original mask
    inv = 1.0 - m
    rad = math.radians(angle)
    ox = round(distance * math.cos(rad))
    oy = round(-distance * math.sin(rad))
    shifted = torch.roll(inv, shifts=(oy, ox), dims=(1, 2))

    if blur_radius > 0:
        sigma = blur_radius / 2.0
        shifted = _gaussian_blur(shifted.unsqueeze(1), blur_radius, sigma).squeeze(1)

    shadow_alpha = (shifted * m * opacity).clamp(0, 1)

    shadow_color = torch.tensor([shadow_r, shadow_g, shadow_b],
                                dtype=image.dtype, device=image.device)
    shadow_layer = shadow_color.view(1, 1, 1, 3).expand(b, h, w, -1)

    alpha = shadow_alpha.unsqueeze(-1)
    result = image * (1.0 - alpha) + shadow_layer * alpha
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "GradientOverlay",
    return_types=("IMAGE",),
    category="image/compositing",
    input_types={"required": {
        "image": ("IMAGE",),
        "start_r": ("FLOAT",), "start_g": ("FLOAT",), "start_b": ("FLOAT",),
        "end_r": ("FLOAT",), "end_g": ("FLOAT",), "end_b": ("FLOAT",),
        "angle": ("FLOAT",),
        "blend_mode": ("STRING",),
        "opacity": ("FLOAT",),
    }},
)
def gradient_overlay(image, start_r=0.0, start_g=0.0, start_b=0.0,
                     end_r=1.0, end_g=1.0, end_b=1.0, angle=0.0,
                     blend_mode="normal", opacity=0.5):
    b, h, w, c = image.shape
    # Build gradient
    grad = _build_linear_gradient(h, w, angle, image.device, image.dtype)  # (H, W, 1)
    start = torch.tensor([start_r, start_g, start_b], dtype=image.dtype, device=image.device)
    end = torch.tensor([end_r, end_g, end_b], dtype=image.dtype, device=image.device)
    grad_color = start * (1.0 - grad) + end * grad  # (H, W, 3)
    grad_layer = grad_color.unsqueeze(0).expand(b, -1, -1, -1)

    blended = _blend_fn(image[..., :3], grad_layer, blend_mode)
    blended = torch.clamp(blended, 0, 1)
    result = image[..., :3] * (1.0 - opacity) + blended * opacity
    if c > 3:
        result = torch.cat([result, image[..., 3:]], dim=-1)
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "ColorOverlay",
    return_types=("IMAGE",),
    category="image/compositing",
    input_types={"required": {
        "image": ("IMAGE",),
        "color_r": ("FLOAT",),
        "color_g": ("FLOAT",),
        "color_b": ("FLOAT",),
        "blend_mode": ("STRING",),
        "opacity": ("FLOAT",),
    }},
)
def color_overlay(image, color_r=1.0, color_g=0.0, color_b=0.0,
                  blend_mode="normal", opacity=0.5):
    color = torch.tensor([color_r, color_g, color_b],
                         dtype=image.dtype, device=image.device)
    color_layer = color.view(1, 1, 1, 3).expand_as(image[..., :3])
    blended = _blend_fn(image[..., :3], color_layer, blend_mode)
    blended = torch.clamp(blended, 0, 1)
    result = image[..., :3] * (1.0 - opacity) + blended * opacity
    if image.shape[-1] > 3:
        result = torch.cat([result, image[..., 3:]], dim=-1)
    return (torch.clamp(result, 0, 1),)


# ---------------------------------------------------------------------------
# Utility Nodes
# ---------------------------------------------------------------------------

@registry.register(
    "ColorImage",
    return_types=("IMAGE",),
    category="image/utility",
    input_types={"required": {
        "width": ("INT",),
        "height": ("INT",),
        "color_r": ("FLOAT",),
        "color_g": ("FLOAT",),
        "color_b": ("FLOAT",),
    }},
)
def color_image(width, height, color_r=1.0, color_g=1.0, color_b=1.0):
    img = torch.zeros(1, height, width, 3, dtype=torch.float32)
    img[..., 0] = color_r
    img[..., 1] = color_g
    img[..., 2] = color_b
    return (torch.clamp(img, 0, 1),)


def _build_linear_gradient(h, w, angle_deg, device, dtype):
    """Build a linear gradient (H, W, 1) from 0 to 1 along *angle_deg*."""
    rad = math.radians(angle_deg)
    # Create coordinate grid normalised to [-1, 1]
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # Project onto gradient direction
    proj = xx * math.cos(rad) + yy * math.sin(rad)
    # Normalise to [0, 1]
    proj = (proj - proj.min()) / (proj.max() - proj.min() + _EPS)
    return proj.unsqueeze(-1)


@registry.register(
    "GradientImage",
    return_types=("IMAGE",),
    category="image/utility",
    input_types={"required": {
        "width": ("INT",),
        "height": ("INT",),
        "start_r": ("FLOAT",), "start_g": ("FLOAT",), "start_b": ("FLOAT",),
        "end_r": ("FLOAT",), "end_g": ("FLOAT",), "end_b": ("FLOAT",),
        "gradient_type": ("STRING",),
        "angle": ("FLOAT",),
        "center_x": ("FLOAT",),
        "center_y": ("FLOAT",),
    }},
)
def gradient_image(width, height, start_r=0.0, start_g=0.0, start_b=0.0,
                   end_r=1.0, end_g=1.0, end_b=1.0, gradient_type="linear",
                   angle=0.0, center_x=0.5, center_y=0.5):
    start = torch.tensor([start_r, start_g, start_b], dtype=torch.float32)
    end = torch.tensor([end_r, end_g, end_b], dtype=torch.float32)

    if gradient_type == "radial":
        ys = torch.linspace(0, 1, height)
        xs = torch.linspace(0, 1, width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dist = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        dist = dist / (dist.max() + _EPS)
        t = dist.unsqueeze(-1)
    else:  # linear
        t = _build_linear_gradient(height, width, angle, torch.device("cpu"), torch.float32)

    img = start * (1.0 - t) + end * t
    return (torch.clamp(img.unsqueeze(0), 0, 1),)


@registry.register(
    "SimpleTextImage",
    return_types=("IMAGE",),
    category="image/utility",
    input_types={"required": {
        "text": ("STRING",),
        "width": ("INT",),
        "height": ("INT",),
        "font_size": ("INT",),
        "color_r": ("FLOAT",), "color_g": ("FLOAT",), "color_b": ("FLOAT",),
        "bg_r": ("FLOAT",), "bg_g": ("FLOAT",), "bg_b": ("FLOAT",),
    }},
)
def simple_text_image(text, width=512, height=128, font_size=32,
                      color_r=1.0, color_g=1.0, color_b=1.0,
                      bg_r=0.0, bg_g=0.0, bg_b=0.0):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # Fallback: return background-colored image with no text
        img = torch.zeros(1, height, width, 3, dtype=torch.float32)
        img[..., 0] = bg_r
        img[..., 1] = bg_g
        img[..., 2] = bg_b
        return (torch.clamp(img, 0, 1),)

    bg_color = (int(bg_r * 255), int(bg_g * 255), int(bg_b * 255))
    fg_color = (int(color_r * 255), int(color_g * 255), int(color_b * 255))
    pil_img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (width - tw) // 2
    ty = (height - th) // 2
    draw.text((tx, ty), text, fill=fg_color, font=font)
    import numpy as np
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # 1, H, W, 3
    return (tensor,)


@registry.register(
    "ExtendCanvas",
    return_types=("IMAGE",),
    category="image/transform",
    input_types={"required": {
        "image": ("IMAGE",),
        "top": ("INT",), "bottom": ("INT",),
        "left": ("INT",), "right": ("INT",),
        "fill_r": ("FLOAT",), "fill_g": ("FLOAT",), "fill_b": ("FLOAT",),
    }},
)
def extend_canvas(image, top=0, bottom=0, left=0, right=0,
                  fill_r=0.0, fill_g=0.0, fill_b=0.0):
    b, h, w, c = image.shape
    new_h = h + top + bottom
    new_w = w + left + right
    result = torch.zeros(b, new_h, new_w, c, dtype=image.dtype, device=image.device)
    fill = torch.tensor([fill_r, fill_g, fill_b], dtype=image.dtype, device=image.device)
    result[..., :3] = fill.view(1, 1, 1, 3)
    if c > 3:
        result[..., 3:] = 0.0
    result[:, top:top + h, left:left + w, :] = image
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "ImageReel",
    return_types=("IMAGE",),
    category="image/utility",
    input_types={"required": {
        "images": ("IMAGE",),
        "columns": ("INT",),
        "rows": ("INT",),
        "spacing": ("INT",),
    }},
)
def image_reel(images, columns=4, rows=0, spacing=4):
    n, ih, iw, c = images.shape
    if rows <= 0:
        rows = max(1, math.ceil(n / max(1, columns)))
    total_w = columns * iw + (columns - 1) * spacing
    total_h = rows * ih + (rows - 1) * spacing
    result = torch.zeros(1, total_h, total_w, c, dtype=images.dtype, device=images.device)
    for idx in range(min(n, rows * columns)):
        row_i = idx // columns
        col_i = idx % columns
        y = row_i * (ih + spacing)
        x = col_i * (iw + spacing)
        result[0, y:y + ih, x:x + iw, :] = images[idx]
    return (result,)


# ---------------------------------------------------------------------------
# Mask Utilities (additions)
# ---------------------------------------------------------------------------

@registry.register(
    "MaskBlur",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "mask": ("MASK",),
        "radius": ("INT",),
    }},
)
def mask_blur(mask, radius=3):
    if radius <= 0:
        return (mask,)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    sigma = radius / 2.0
    blurred = _gaussian_blur(mask.unsqueeze(1), radius, sigma).squeeze(1)
    return (blurred.clamp(0, 1),)


@registry.register(
    "MaskExpand",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "mask": ("MASK",),
        "pixels": ("INT",),
    }},
)
def mask_expand(mask, pixels=5):
    if pixels <= 0:
        return (mask,)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    m = mask.unsqueeze(1)
    ksize = pixels * 2 + 1
    kernel = torch.ones(1, 1, ksize, ksize, dtype=mask.dtype, device=mask.device)
    padded = F.pad(m, (pixels, pixels, pixels, pixels), mode="constant", value=0.0)
    result = F.conv2d(padded, kernel, padding=0)
    result = (result > 0.5).float()
    return (result.squeeze(1),)


@registry.register(
    "MaskErode",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "mask": ("MASK",),
        "pixels": ("INT",),
    }},
)
def mask_erode(mask, pixels=5):
    if pixels <= 0:
        return (mask,)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    m = mask.unsqueeze(1)
    ksize = pixels * 2 + 1
    kernel = torch.ones(1, 1, ksize, ksize, dtype=mask.dtype, device=mask.device)
    # Erode = invert, dilate (with boundary=1 so edges erode), invert
    inv = 1.0 - m
    padded = F.pad(inv, (pixels, pixels, pixels, pixels), mode="constant", value=1.0)
    dilated = F.conv2d(padded, kernel, padding=0)
    result = 1.0 - (dilated > 0.5).float()
    return (result.squeeze(1),)


@registry.register(
    "CropByMask",
    return_types=("IMAGE", "STITCH_DATA"),
    return_names=("IMAGE", "STITCH_DATA"),
    category="image/compositing",
    input_types={"required": {
        "image": ("IMAGE",),
        "mask": ("MASK",),
        "padding": ("INT",),
    }},
)
def crop_by_mask(image, mask, padding=16):
    b, h, w, c = image.shape
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    # Use first mask in batch to find bbox
    m0 = m[0]
    nonzero = m0.nonzero(as_tuple=False)
    if nonzero.shape[0] == 0:
        # Empty mask: return full image
        stitch = {"x": 0, "y": 0, "orig_h": h, "orig_w": w}
        return (image, stitch)

    y_min = nonzero[:, 0].min().item()
    y_max = nonzero[:, 0].max().item()
    x_min = nonzero[:, 1].min().item()
    x_max = nonzero[:, 1].max().item()

    # Apply padding
    y_min = max(0, y_min - padding)
    y_max = min(h - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w - 1, x_max + padding)

    cropped = image[:, y_min:y_max + 1, x_min:x_max + 1, :].contiguous()
    stitch = {"x": x_min, "y": y_min, "orig_h": h, "orig_w": w}
    return (cropped, stitch)


@registry.register(
    "RestoreCropBox",
    return_types=("IMAGE",),
    category="image/compositing",
    input_types={"required": {
        "cropped_image": ("IMAGE",),
        "stitch_data": ("STITCH_DATA",),
        "background": ("IMAGE",),
    }},
)
def restore_crop_box(cropped_image, stitch_data, background):
    output = background.clone()
    x = stitch_data["x"]
    y = stitch_data["y"]
    _, ch, cw, _ = cropped_image.shape
    b, bh, bw, _ = output.shape

    # Clamp to bounds
    rw = min(cw, bw - x)
    rh = min(ch, bh - y)
    if rw > 0 and rh > 0:
        output[:, y:y + rh, x:x + rw, :] = cropped_image[:, :rh, :rw, :]
    return (torch.clamp(output, 0, 1),)


__all__ = [
    "image_blend_advanced",
    "exposure",
    "color_temperature",
    "lut_apply",
    "auto_adjust",
    "negative",
    "add_grain",
    "channel_shake",
    "drop_shadow",
    "inner_glow",
    "inner_shadow",
    "gradient_overlay",
    "color_overlay",
    "color_image",
    "gradient_image",
    "simple_text_image",
    "extend_canvas",
    "image_reel",
    "mask_blur",
    "mask_expand",
    "mask_erode",
    "crop_by_mask",
    "restore_crop_box",
]
