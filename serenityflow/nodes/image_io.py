"""Image I/O nodes -- LoadImage, SaveImage, PreviewImage, ImageScale."""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from serenityflow.nodes.registry import registry

log = logging.getLogger(__name__)

# Module-level counter for output filenames
_output_counter: dict[str, int] = {}


@registry.register(
    "LoadImage",
    return_types=("IMAGE", "MASK"),
    category="image",
    input_types={"required": {"image": ("STRING",)}},
)
def load_image(image):
    from serenityflow.bridge.model_paths import get_model_paths

    # Resolve path: absolute or relative to input dir
    if os.path.isabs(image):
        filepath = image
    else:
        paths = get_model_paths()
        input_dir = os.path.join(paths.base_dir, "input")
        filepath = os.path.join(input_dir, image)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")

    img = Image.open(filepath)
    img = img.convert("RGBA")
    img_array = np.array(img).astype(np.float32) / 255.0

    # Split RGB and alpha -> BHWC
    rgb = torch.from_numpy(img_array[:, :, :3]).unsqueeze(0)  # [1, H, W, 3]
    alpha = torch.from_numpy(img_array[:, :, 3]).unsqueeze(0)  # [1, H, W]
    mask = 1.0 - alpha  # ComfyUI: mask=1 means masked area

    return (rgb, mask)


@registry.register(
    "SaveImage",
    return_types=(),
    category="image",
    is_output=True,
    input_types={"required": {"images": ("IMAGE",), "filename_prefix": ("STRING",)}},
)
def save_image(images, filename_prefix="SerenityFlow"):
    from serenityflow.bridge.model_paths import get_model_paths

    paths = get_model_paths()
    output_dir = os.path.join(paths.base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    counter = _output_counter.get(filename_prefix, 0)

    for i in range(images.shape[0]):
        img_tensor = images[i]  # [H, W, C] BHWC already sliced
        img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        filename = f"{filename_prefix}_{counter + i:05d}.png"
        filepath = os.path.join(output_dir, filename)
        pil_img.save(filepath, compress_level=4)
        log.info("Saved: %s", filepath)
        results.append({"filename": filename, "subfolder": "", "type": "output"})

    _output_counter[filename_prefix] = counter + images.shape[0]
    return {"ui": {"images": results}}


@registry.register(
    "PreviewImage",
    return_types=(),
    category="image",
    is_output=True,
    input_types={"required": {"images": ("IMAGE",)}},
)
def preview_image(images):
    from serenityflow.bridge.model_paths import get_model_paths

    paths = get_model_paths()
    temp_dir = os.path.join(paths.base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    results = []
    for i in range(images.shape[0]):
        img_tensor = images[i]
        img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        filename = f"preview_{id(images)}_{i}.png"
        filepath = os.path.join(temp_dir, filename)
        pil_img.save(filepath)
        results.append({"filename": filename, "subfolder": "", "type": "temp"})

    return {"ui": {"images": results}}


@registry.register(
    "ImageScale",
    return_types=("IMAGE",),
    category="image",
    input_types={"required": {
        "image": ("IMAGE",), "upscale_method": ("STRING",),
        "width": ("INT",), "height": ("INT",), "crop": ("STRING",),
    }},
)
def image_scale(image, upscale_method, width, height, crop="disabled"):
    # BHWC -> BCHW for interpolation
    img = image.permute(0, 3, 1, 2)
    mode = "bilinear" if upscale_method not in ("nearest", "nearest-exact") else "nearest"
    align = None if mode == "nearest" else False
    resized = F.interpolate(img, size=(height, width), mode=mode, align_corners=align)
    return (resized.permute(0, 2, 3, 1),)  # BCHW -> BHWC


@registry.register(
    "ImageScaleBy",
    return_types=("IMAGE",),
    category="image",
    input_types={"required": {
        "image": ("IMAGE",), "upscale_method": ("STRING",), "scale_by": ("FLOAT",),
    }},
)
def image_scale_by(image, upscale_method, scale_by):
    _, h, w, _ = image.shape
    new_h = round(h * scale_by)
    new_w = round(w * scale_by)
    return image_scale(image, upscale_method, new_w, new_h, "disabled")


@registry.register(
    "ImageBatch",
    return_types=("IMAGE",),
    category="image",
    input_types={"required": {
        "image1": ("IMAGE",), "image2": ("IMAGE",),
    }},
)
def image_batch(image1, image2):
    # Resize image2 to match image1 spatial dims if needed
    if image1.shape[1:3] != image2.shape[1:3]:
        img2 = image2.permute(0, 3, 1, 2)
        img2 = F.interpolate(img2, size=image1.shape[1:3], mode="bilinear", align_corners=False)
        image2 = img2.permute(0, 2, 3, 1)
    return (torch.cat([image1, image2], dim=0),)


@registry.register(
    "ImageInvert",
    return_types=("IMAGE",),
    category="image",
    input_types={"required": {"image": ("IMAGE",)}},
)
def image_invert(image):
    return (1.0 - image,)


@registry.register(
    "ImagePadForOutpaint",
    return_types=("IMAGE", "MASK"),
    category="image",
    input_types={"required": {
        "image": ("IMAGE",),
        "left": ("INT",), "top": ("INT",),
        "right": ("INT",), "bottom": ("INT",),
        "feathering": ("INT",),
    }},
)
def image_pad_for_outpaint(image, left, top, right, bottom, feathering=40):
    b, h, w, c = image.shape
    new_h = h + top + bottom
    new_w = w + left + right
    # Pad image with zeros
    padded = torch.zeros(b, new_h, new_w, c, dtype=image.dtype, device=image.device)
    padded[:, top:top + h, left:left + w, :] = image
    # Mask: 1.0 = needs generation, 0.0 = keep original
    mask = torch.ones(b, new_h, new_w, dtype=torch.float32, device=image.device)
    mask[:, top:top + h, left:left + w] = 0.0
    # Feathering on the mask boundary
    if feathering > 0:
        for i in range(feathering):
            alpha = (i + 1) / feathering
            # Top edge
            if top + i < new_h:
                mask[:, top + i, left:left + w] = max(mask[:, top + i, left:left + w].max().item(), 1.0 - alpha)
            # Bottom edge
            if top + h - 1 - i >= 0:
                mask[:, top + h - 1 - i, left:left + w] = max(mask[:, top + h - 1 - i, left:left + w].max().item(), 1.0 - alpha)
            # Left edge
            if left + i < new_w:
                mask[:, top:top + h, left + i] = torch.maximum(mask[:, top:top + h, left + i], torch.tensor(1.0 - alpha))
            # Right edge
            if left + w - 1 - i >= 0:
                mask[:, top:top + h, left + w - 1 - i] = torch.maximum(mask[:, top:top + h, left + w - 1 - i], torch.tensor(1.0 - alpha))
    return (padded, mask)
