"""Image operation nodes -- crop, flip, rotate, composite, sharpen, blur, etc."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from serenityflow.nodes.registry import registry


@registry.register(
    "ImageCrop",
    return_types=("IMAGE",),
    category="image/transform",
    input_types={"required": {
        "image": ("IMAGE",),
        "width": ("INT",), "height": ("INT",),
        "x": ("INT",), "y": ("INT",),
    }},
)
def image_crop(image, width, height, x, y):
    b, h, w, c = image.shape
    # Clamp to image bounds
    x2 = min(x + width, w)
    y2 = min(y + height, h)
    x = max(0, x)
    y = max(0, y)
    return (image[:, y:y2, x:x2, :].contiguous(),)


@registry.register(
    "ImageCompositeMasked",
    return_types=("IMAGE",),
    category="image",
    input_types={
        "required": {
            "destination": ("IMAGE",),
            "source": ("IMAGE",),
            "x": ("INT",), "y": ("INT",),
        },
        "optional": {"mask": ("MASK",)},
    },
)
def image_composite_masked(destination, source, x, y, mask=None):
    output = destination.clone()
    b, dh, dw, c = output.shape
    _, sh, sw, _ = source.shape

    # Region that overlaps
    sx, sy = max(0, -x), max(0, -y)
    dx, dy = max(0, x), max(0, y)
    rw = min(sw - sx, dw - dx)
    rh = min(sh - sy, dh - dy)
    if rw <= 0 or rh <= 0:
        return (output,)

    src_region = source[:, sy:sy + rh, sx:sx + rw, :]
    if mask is not None:
        # Expand mask to match region
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        # Crop mask to source region
        m = m[:, sy:sy + rh, sx:sx + rw]
        # Broadcast batch
        if m.shape[0] < b:
            m = m.expand(b, -1, -1)
        m = m.unsqueeze(-1)  # BHW -> BHWC-broadcastable
        output[:, dy:dy + rh, dx:dx + rw, :] = (
            output[:, dy:dy + rh, dx:dx + rw, :] * (1.0 - m) + src_region * m
        )
    else:
        output[:, dy:dy + rh, dx:dx + rw, :] = src_region
    return (output,)


@registry.register(
    "ImageFlip",
    return_types=("IMAGE",),
    category="image/transform",
    input_types={"required": {
        "image": ("IMAGE",),
        "flip_method": ("STRING",),
    }},
)
def image_flip(image, flip_method="horizontal"):
    if flip_method == "horizontal":
        return (torch.flip(image, [2]),)  # flip W
    else:
        return (torch.flip(image, [1]),)  # flip H


@registry.register(
    "ImageRotate",
    return_types=("IMAGE",),
    category="image/transform",
    input_types={"required": {
        "image": ("IMAGE",),
        "rotation": ("STRING",),
    }},
)
def image_rotate(image, rotation="none"):
    k = {"none": 0, "90 degrees": 1, "180 degrees": 2, "270 degrees": 3}.get(rotation, 0)
    if k > 0:
        # BHWC: rotate dims 1 (H) and 2 (W)
        image = torch.rot90(image, k=k, dims=[1, 2])
    return (image,)


@registry.register(
    "ImageSharpen",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "sharpen_radius": ("INT",),
        "sigma": ("FLOAT",),
        "alpha": ("FLOAT",),
    }},
)
def image_sharpen(image, sharpen_radius=1, sigma=1.0, alpha=1.0):
    if sharpen_radius == 0 or alpha == 0.0:
        return (image,)
    # Unsharp mask: sharp = original + alpha * (original - blurred)
    img = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
    ksize = sharpen_radius * 2 + 1
    # Create Gaussian kernel
    x = torch.arange(ksize, dtype=torch.float32, device=image.device) - sharpen_radius
    kernel_1d = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.expand(img.shape[1], 1, ksize, ksize)
    pad = sharpen_radius
    blurred = F.conv2d(
        F.pad(img, (pad, pad, pad, pad), mode="reflect"),
        kernel_2d, groups=img.shape[1],
    )
    result = img + alpha * (img - blurred)
    return (torch.clamp(result.permute(0, 2, 3, 1), 0, 1),)


@registry.register(
    "ImageBlur",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "blur_radius": ("INT",),
        "sigma": ("FLOAT",),
    }},
)
def image_blur(image, blur_radius=1, sigma=1.0):
    if blur_radius == 0:
        return (image,)
    img = image.permute(0, 3, 1, 2)
    ksize = blur_radius * 2 + 1
    x = torch.arange(ksize, dtype=torch.float32, device=image.device) - blur_radius
    kernel_1d = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.expand(img.shape[1], 1, ksize, ksize)
    pad = blur_radius
    blurred = F.conv2d(
        F.pad(img, (pad, pad, pad, pad), mode="reflect"),
        kernel_2d, groups=img.shape[1],
    )
    return (torch.clamp(blurred.permute(0, 2, 3, 1), 0, 1),)


@registry.register(
    "ImageQuantize",
    return_types=("IMAGE",),
    category="image/postprocessing",
    input_types={"required": {
        "image": ("IMAGE",),
        "colors": ("INT",),
    }},
)
def image_quantize(image, colors=256):
    if colors < 2:
        colors = 2
    levels = colors - 1
    result = (image * levels).round() / levels
    return (torch.clamp(result, 0, 1),)


@registry.register(
    "ImageToMask",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "image": ("IMAGE",),
        "channel": ("STRING",),
    }},
)
def image_to_mask(image, channel="red"):
    ch_map = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
    idx = ch_map.get(channel, 0)
    if idx >= image.shape[3]:
        # If alpha requested but image is RGB, return ones
        return (torch.ones(image.shape[0], image.shape[1], image.shape[2],
                          dtype=torch.float32, device=image.device),)
    return (image[:, :, :, idx],)


@registry.register(
    "MaskToImage",
    return_types=("IMAGE",),
    category="mask",
    input_types={"required": {"mask": ("MASK",)}},
)
def mask_to_image(mask):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    # BHW -> BHWC (3-channel grayscale)
    return (mask.unsqueeze(-1).expand(-1, -1, -1, 3),)


@registry.register(
    "SplitImageWithAlpha",
    return_types=("IMAGE", "MASK"),
    category="mask",
    input_types={"required": {"image": ("IMAGE",)}},
)
def split_image_with_alpha(image):
    if image.shape[3] >= 4:
        rgb = image[:, :, :, :3]
        alpha = image[:, :, :, 3]
    else:
        rgb = image
        alpha = torch.ones(image.shape[0], image.shape[1], image.shape[2],
                          dtype=torch.float32, device=image.device)
    return (rgb, alpha)


@registry.register(
    "JoinImageWithAlpha",
    return_types=("IMAGE",),
    category="mask",
    input_types={"required": {
        "image": ("IMAGE",),
        "alpha": ("MASK",),
    }},
)
def join_image_with_alpha(image, alpha):
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(0)
    # Resize alpha to match image spatial dims if needed
    if alpha.shape[1:] != image.shape[1:3]:
        alpha = F.interpolate(
            alpha.unsqueeze(1),
            size=image.shape[1:3],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return (torch.cat([image[:, :, :, :3], alpha.unsqueeze(-1)], dim=3),)


@registry.register(
    "RebatchImages",
    return_types=("IMAGE",),
    category="image/batch",
    input_types={"required": {
        "images": ("IMAGE",),
        "batch_size": ("INT",),
    }},
)
def rebatch_images(images, batch_size=1):
    # Split into chunks of batch_size, return first chunk
    # In ComfyUI this returns a list; here we just reshape
    if batch_size >= images.shape[0]:
        return (images,)
    return (images[:batch_size],)


@registry.register(
    "ImageFromBatch",
    return_types=("IMAGE",),
    category="image/batch",
    input_types={"required": {
        "image": ("IMAGE",),
        "batch_index": ("INT",),
        "length": ("INT",),
    }},
)
def image_from_batch(image, batch_index=0, length=1):
    end = min(batch_index + length, image.shape[0])
    batch_index = max(0, batch_index)
    return (image[batch_index:end],)


@registry.register(
    "RepeatImageBatch",
    return_types=("IMAGE",),
    category="image/batch",
    input_types={"required": {
        "image": ("IMAGE",),
        "amount": ("INT",),
    }},
)
def repeat_image_batch(image, amount=1):
    return (image.repeat(amount, 1, 1, 1),)


@registry.register(
    "ImageScaleToTotalPixels",
    return_types=("IMAGE",),
    category="image/transform",
    input_types={"required": {
        "image": ("IMAGE",),
        "upscale_method": ("STRING",),
        "megapixels": ("FLOAT",),
    }},
)
def image_scale_to_total_pixels(image, upscale_method="bilinear", megapixels=1.0):
    _, h, w, _ = image.shape
    total = megapixels * 1024 * 1024
    scale = (total / (h * w)) ** 0.5
    new_w = round(w * scale)
    new_h = round(h * scale)
    img = image.permute(0, 3, 1, 2)
    mode = "nearest" if upscale_method in ("nearest", "nearest-exact") else "bilinear"
    align = None if mode == "nearest" else False
    resized = F.interpolate(img, size=(new_h, new_w), mode=mode, align_corners=align)
    return (resized.permute(0, 2, 3, 1),)


@registry.register(
    "BatchImagesNode",
    return_types=("IMAGE",),
    category="image/batch",
    input_types={
        "required": {"image1": ("IMAGE",)},
        "optional": {"image2": ("IMAGE",), "image3": ("IMAGE",)},
    },
)
def batch_images_node(image1, image2=None, image3=None):
    images = [image1]
    for img in (image2, image3):
        if img is not None:
            if img.shape[1:3] != image1.shape[1:3]:
                t = img.permute(0, 3, 1, 2)
                t = F.interpolate(t, size=image1.shape[1:3], mode="bilinear", align_corners=False)
                img = t.permute(0, 2, 3, 1)
            images.append(img)
    return (torch.cat(images, dim=0),)


@registry.register(
    "ImageStitch",
    return_types=("IMAGE",),
    category="image/transform",
    input_types={"required": {
        "image1": ("IMAGE",), "image2": ("IMAGE",),
        "direction": ("STRING",),
    }},
)
def image_stitch(image1, image2, direction="horizontal"):
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape
    img2 = image2.permute(0, 3, 1, 2)
    if direction == "horizontal":
        # Resize image2 height to match image1
        if h2 != h1:
            new_w2 = round(w2 * h1 / h2)
            img2 = F.interpolate(img2, size=(h1, new_w2), mode="bilinear", align_corners=False)
        img2 = img2.permute(0, 2, 3, 1)
        return (torch.cat([image1, img2], dim=2),)
    else:
        # Resize image2 width to match image1
        if w2 != w1:
            new_h2 = round(h2 * w1 / w2)
            img2 = F.interpolate(img2, size=(new_h2, w1), mode="bilinear", align_corners=False)
        img2 = img2.permute(0, 2, 3, 1)
        return (torch.cat([image1, img2], dim=1),)


@registry.register(
    "ResizeImageMaskNode",
    return_types=("IMAGE", "MASK"),
    return_names=("IMAGE", "MASK"),
    category="image/transform",
    input_types={"required": {
        "image": ("IMAGE",), "mask": ("MASK",),
        "width": ("INT",), "height": ("INT",),
        "upscale_method": ("STRING",),
    }},
)
def resize_image_mask_node(image, mask, width, height, upscale_method="bilinear"):
    mode = "nearest" if upscale_method in ("nearest", "nearest-exact") else "bilinear"
    align = None if mode == "nearest" else False
    # Resize image (BHWC -> BCHW -> resize -> BHWC)
    img = image.permute(0, 3, 1, 2)
    img = F.interpolate(img, size=(height, width), mode=mode, align_corners=align)
    img = img.permute(0, 2, 3, 1)
    # Resize mask (BHW -> B1HW -> resize -> BHW)
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    m = m.unsqueeze(1)
    m = F.interpolate(m, size=(height, width), mode=mode, align_corners=align)
    m = m.squeeze(1)
    return (img, m)


@registry.register(
    "ResizeAndPadImage",
    return_types=("IMAGE",),
    category="image/transform",
    input_types={"required": {
        "image": ("IMAGE",), "width": ("INT",), "height": ("INT",),
    }},
)
def resize_and_pad_image(image, width, height):
    _, h, w, _ = image.shape
    scale = min(width / w, height / h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    img = image.permute(0, 3, 1, 2)
    img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    # Pad to exact target size
    pad_left = (width - new_w) // 2
    pad_right = width - new_w - pad_left
    pad_top = (height - new_h) // 2
    pad_bottom = height - new_h - pad_top
    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
    return (img.permute(0, 2, 3, 1),)


@registry.register(
    "ImageScaleToMaxDimension",
    return_types=("IMAGE",),
    category="image/transform",
    input_types={"required": {
        "image": ("IMAGE",),
        "max_dimension": ("INT",),
        "upscale_method": ("STRING",),
    }},
)
def image_scale_to_max_dimension(image, max_dimension=1024, upscale_method="bilinear"):
    _, h, w, _ = image.shape
    scale = max_dimension / max(h, w)
    new_w = round(w * scale)
    new_h = round(h * scale)
    img = image.permute(0, 3, 1, 2)
    mode = "nearest" if upscale_method in ("nearest", "nearest-exact") else "bilinear"
    align = None if mode == "nearest" else False
    resized = F.interpolate(img, size=(new_h, new_w), mode=mode, align_corners=align)
    return (resized.permute(0, 2, 3, 1),)


@registry.register(
    "ImageBatchMulti",
    return_types=("IMAGE",),
    category="image/batch",
    input_types={
        "required": {"image1": ("IMAGE",)},
        "optional": {"image2": ("IMAGE",)},
    },
)
def image_batch_multi(image1, image2=None):
    if image2 is None:
        return (image1,)
    if image2.shape[1:3] != image1.shape[1:3]:
        img2 = image2.permute(0, 3, 1, 2)
        img2 = F.interpolate(img2, size=image1.shape[1:3], mode="bilinear", align_corners=False)
        image2 = img2.permute(0, 2, 3, 1)
    return (torch.cat([image1, image2], dim=0),)
