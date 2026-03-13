"""Mask operation nodes -- invert, composite, feather, grow, threshold, solid."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from serenityflow.nodes.registry import registry


@registry.register(
    "InvertMask",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {"mask": ("MASK",)}},
)
def invert_mask(mask):
    return (1.0 - mask,)


@registry.register(
    "CropMask",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "mask": ("MASK",),
        "x": ("INT",), "y": ("INT",),
        "width": ("INT",), "height": ("INT",),
    }},
)
def crop_mask(mask, x, y, width, height):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    _, h, w = mask.shape
    x2 = min(x + width, w)
    y2 = min(y + height, h)
    x = max(0, x)
    y = max(0, y)
    return (mask[:, y:y2, x:x2].contiguous(),)


@registry.register(
    "MaskComposite",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "destination": ("MASK",),
        "source": ("MASK",),
        "x": ("INT",), "y": ("INT",),
        "operation": ("STRING",),
    }},
)
def mask_composite(destination, source, x, y, operation="add"):
    if destination.ndim == 2:
        destination = destination.unsqueeze(0)
    if source.ndim == 2:
        source = source.unsqueeze(0)
    output = destination.clone()
    _, dh, dw = output.shape
    _, sh, sw = source.shape

    # Compute overlap region
    sx, sy = max(0, -x), max(0, -y)
    dx, dy = max(0, x), max(0, y)
    rw = min(sw - sx, dw - dx)
    rh = min(sh - sy, dh - dy)
    if rw <= 0 or rh <= 0:
        return (output,)

    dst_region = output[:, dy:dy + rh, dx:dx + rw]
    src_region = source[:, sy:sy + rh, sx:sx + rw]

    if operation == "multiply":
        output[:, dy:dy + rh, dx:dx + rw] = dst_region * src_region
    elif operation == "add":
        output[:, dy:dy + rh, dx:dx + rw] = dst_region + src_region
    elif operation == "subtract":
        output[:, dy:dy + rh, dx:dx + rw] = dst_region - src_region
    elif operation == "and":
        output[:, dy:dy + rh, dx:dx + rw] = torch.min(dst_region, src_region)
    elif operation == "or":
        output[:, dy:dy + rh, dx:dx + rw] = torch.max(dst_region, src_region)
    elif operation == "xor":
        output[:, dy:dy + rh, dx:dx + rw] = torch.abs(dst_region - src_region)

    return (torch.clamp(output, 0, 1),)


@registry.register(
    "FeatherMask",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "mask": ("MASK",),
        "left": ("INT",), "top": ("INT",),
        "right": ("INT",), "bottom": ("INT",),
    }},
)
def feather_mask(mask, left=0, top=0, right=0, bottom=0):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    output = mask.clone()
    _, h, w = output.shape

    for i in range(left):
        if i < w:
            output[:, :, i] *= i / left
    for i in range(right):
        if w - 1 - i >= 0:
            output[:, :, w - 1 - i] *= i / right
    for i in range(top):
        if i < h:
            output[:, i, :] *= i / top
    for i in range(bottom):
        if h - 1 - i >= 0:
            output[:, h - 1 - i, :] *= i / bottom

    return (torch.clamp(output, 0, 1),)


@registry.register(
    "GrowMask",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "mask": ("MASK",),
        "expand": ("INT",),
        "tapered_corners": ("BOOLEAN",),
    }},
)
def grow_mask(mask, expand=0, tapered_corners=True):
    if expand == 0:
        return (mask,)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    abs_expand = abs(expand)
    # Add channel dim for pooling: BHW -> B1HW
    m = mask.unsqueeze(1)
    pad = abs_expand
    m_padded = F.pad(m, (pad, pad, pad, pad), mode="constant", value=0.0)

    ksize = abs_expand * 2 + 1
    if tapered_corners:
        # Circular kernel
        y = torch.arange(ksize, device=mask.device).float() - abs_expand
        x = torch.arange(ksize, device=mask.device).float() - abs_expand
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        kernel = ((xx ** 2 + yy ** 2) <= abs_expand ** 2).float()
    else:
        kernel = torch.ones(ksize, ksize, device=mask.device)

    kernel = kernel.unsqueeze(0).unsqueeze(0)  # 1,1,K,K

    if expand > 0:
        # Dilate: max pool equivalent via conv with >= threshold
        result = F.conv2d(m_padded, kernel, padding=0)
        result = (result > 0.5).float()
    else:
        # Erode: min pool by inverting, dilating, inverting
        inv = 1.0 - m_padded
        result = F.conv2d(inv, kernel, padding=0)
        result = 1.0 - (result > 0.5).float()

    return (result.squeeze(1),)


@registry.register(
    "ThresholdMask",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "mask": ("MASK",),
        "value": ("FLOAT",),
    }},
)
def threshold_mask(mask, value=0.5):
    return ((mask >= value).float(),)


@registry.register(
    "SolidMask",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "value": ("FLOAT",),
        "width": ("INT",), "height": ("INT",),
    }},
)
def solid_mask(value, width, height):
    return (torch.full((1, height, width), value, dtype=torch.float32),)


@registry.register(
    "RebatchMasks",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "masks": ("MASK",),
        "batch_size": ("INT",),
    }},
)
def rebatch_masks(masks, batch_size=1):
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if batch_size >= masks.shape[0]:
        return (masks,)
    return (masks[:batch_size],)


@registry.register(
    "MaskFromBatch",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "mask": ("MASK",),
        "batch_index": ("INT",),
        "length": ("INT",),
    }},
)
def mask_from_batch(mask, batch_index=0, length=1):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    end = min(batch_index + length, mask.shape[0])
    batch_index = max(0, batch_index)
    return (mask[batch_index:end],)
