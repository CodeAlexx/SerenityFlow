"""Upscale & Hires Fix nodes -- bislerp latent upscale, UltimateSDUpscale, latent compositing."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from serenityflow.bridge.types import unwrap_latent, wrap_latent
from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# Bislerp -- spherical linear interpolation for latent upscaling
# ---------------------------------------------------------------------------

def _slerp_1d(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between tensors along last dim."""
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp(min=1e-8)
    result = (torch.sin((1.0 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b
    # Fallback to lerp when omega is very small (parallel vectors)
    mask = omega.abs() < 1e-5
    if mask.any():
        lerped = a * (1.0 - t) + b * t
        result = torch.where(mask, lerped, result)
    return result


def bislerp(input_tensor: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Bilinear slerp -- interpolate a BCHW tensor using slerp along both axes.

    Two-pass approach: slerp horizontally to target width, then vertically to
    target height.  This gives higher-quality latent interpolation than standard
    bilinear because it preserves direction (useful for diffusion latents that
    encode magnitude + direction information).
    """
    B, C, H, W = input_tensor.shape
    if H == height and W == width:
        return input_tensor

    device = input_tensor.device
    dtype = input_tensor.dtype

    # --- Horizontal pass (W -> width) ---
    if W != width:
        # Map target x coords to source coords
        x_coords = torch.linspace(0, W - 1, width, device=device, dtype=dtype)
        x0 = x_coords.long().clamp(0, W - 2)
        x1 = (x0 + 1).clamp(max=W - 1)
        tx = (x_coords - x0.float()).clamp(0.0, 1.0)

        # Gather left/right columns -- shape [B, C, H, width]
        left = input_tensor[:, :, :, x0]   # B, C, H, width
        right = input_tensor[:, :, :, x1]  # B, C, H, width

        # Reshape for slerp: merge B,H into batch, slerp along C
        # left:  [B, C, H, width] -> [B*H*width, C]
        left_flat = left.permute(0, 2, 3, 1).reshape(-1, C)
        right_flat = right.permute(0, 2, 3, 1).reshape(-1, C)
        # tx: [width] -> broadcast to [B*H*width]
        tx_flat = tx.unsqueeze(0).unsqueeze(0).expand(B, H, width).reshape(-1)

        # Per-element slerp
        result_flat = _slerp_element(left_flat, right_flat, tx_flat)
        h_result = result_flat.reshape(B, H, width, C).permute(0, 3, 1, 2)
    else:
        h_result = input_tensor

    # --- Vertical pass (H -> height) ---
    if H != height:
        y_coords = torch.linspace(0, H - 1, height, device=device, dtype=dtype)
        y0 = y_coords.long().clamp(0, H - 2)
        y1 = (y0 + 1).clamp(max=H - 1)
        ty = (y_coords - y0.float()).clamp(0.0, 1.0)

        top = h_result[:, :, y0, :]     # B, C, height, cur_W
        bottom = h_result[:, :, y1, :]   # B, C, height, cur_W

        cur_W = h_result.shape[3]
        top_flat = top.permute(0, 2, 3, 1).reshape(-1, C)
        bottom_flat = bottom.permute(0, 2, 3, 1).reshape(-1, C)
        ty_flat = ty.unsqueeze(0).unsqueeze(-1).expand(B, height, cur_W).reshape(-1)

        result_flat = _slerp_element(top_flat, bottom_flat, ty_flat)
        v_result = result_flat.reshape(B, height, cur_W, C).permute(0, 3, 1, 2)
    else:
        v_result = h_result

    return v_result


def _slerp_element(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Element-wise slerp with per-row t values. a, b: [N, C], t: [N]."""
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp(min=1e-8)
    t = t.unsqueeze(-1)  # [N, 1]
    result = (torch.sin((1.0 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b
    # Fallback to lerp for nearly-parallel vectors
    mask = omega.abs() < 1e-5
    if mask.any():
        lerped = a * (1.0 - t) + b * t
        result = torch.where(mask, lerped, result)
    return result


# ---------------------------------------------------------------------------
# Latent upscale with bislerp support
# ---------------------------------------------------------------------------

def _upscale_latent(latent: torch.Tensor, method: str, h: int, w: int) -> torch.Tensor:
    """Upscale BCHW latent tensor using the given method."""
    if method == "bislerp":
        return bislerp(latent, w, h)
    mode = "nearest" if method in ("nearest", "nearest-exact") else "bilinear"
    align = None if mode == "nearest" else False
    return F.interpolate(latent, size=(h, w), mode=mode, align_corners=align)


@registry.register(
    "LatentUpscaleBislerp",
    return_types=("LATENT",),
    category="latent/upscale",
    input_types={"required": {
        "samples": ("LATENT",),
        "upscale_method": ("STRING",),
        "width": ("INT",),
        "height": ("INT",),
    }},
)
def latent_upscale_bislerp(samples, upscale_method="bislerp", width=1024, height=1024):
    """Latent upscale with bislerp/bilinear/nearest support."""
    latent = unwrap_latent(samples)
    target_h = height // 8
    target_w = width // 8
    result = _upscale_latent(latent, upscale_method, target_h, target_w)
    return (wrap_latent(result),)


@registry.register(
    "LatentUpscaleByBislerp",
    return_types=("LATENT",),
    category="latent/upscale",
    input_types={"required": {
        "samples": ("LATENT",),
        "upscale_method": ("STRING",),
        "scale_by": ("FLOAT",),
    }},
)
def latent_upscale_by_bislerp(samples, upscale_method="bislerp", scale_by=2.0):
    """Scale LATENT by factor with bislerp/bilinear/nearest support."""
    latent = unwrap_latent(samples)
    _, _, h, w = latent.shape
    new_h = round(h * scale_by)
    new_w = round(w * scale_by)
    result = _upscale_latent(latent, upscale_method, new_h, new_w)
    return (wrap_latent(result),)


# ---------------------------------------------------------------------------
# LatentCompositeMasked
# ---------------------------------------------------------------------------

@registry.register(
    "LatentCompositeMasked",
    return_types=("LATENT",),
    category="latent",
    input_types={
        "required": {
            "destination": ("LATENT",),
            "source": ("LATENT",),
            "x": ("INT",),
            "y": ("INT",),
        },
        "optional": {
            "mask": ("MASK",),
        },
    },
)
def latent_composite_masked(destination, source, x=0, y=0, mask=None):
    """Composite source latent onto destination using an optional mask.

    Coordinates are in pixel space (divided by 8 internally for latent space).
    """
    dest = unwrap_latent(destination).clone()
    src = unwrap_latent(source)
    lx, ly = x // 8, y // 8
    _, _, sh, sw = src.shape
    _, _, dh, dw = dest.shape

    # Compute overlap region
    sx_start = max(0, -lx)
    sy_start = max(0, -ly)
    dx_start = max(0, lx)
    dy_start = max(0, ly)
    rw = min(sw - sx_start, dw - dx_start)
    rh = min(sh - sy_start, dh - dy_start)
    if rw <= 0 or rh <= 0:
        return (wrap_latent(dest),)

    src_region = src[:, :, sy_start:sy_start + rh, sx_start:sx_start + rw]

    if mask is not None:
        # Mask is [B, H, W] in pixel space -- downsample to latent space
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        # Resize mask to latent spatial dims of source
        m = F.interpolate(
            m.unsqueeze(1), size=(sh, sw), mode="bilinear", align_corners=False,
        ).squeeze(1)
        # Crop mask to overlap region
        m = m[:, sy_start:sy_start + rh, sx_start:sx_start + rw]
        # Broadcast to channel dim: [B, 1, rh, rw]
        m = m.unsqueeze(1)
        dest_region = dest[:, :, dy_start:dy_start + rh, dx_start:dx_start + rw]
        dest[:, :, dy_start:dy_start + rh, dx_start:dx_start + rw] = (
            dest_region * (1.0 - m) + src_region * m
        )
    else:
        dest[:, :, dy_start:dy_start + rh, dx_start:dx_start + rw] = src_region

    return (wrap_latent(dest),)


# ---------------------------------------------------------------------------
# UltimateSDUpscale -- tiled img2img upscale
# ---------------------------------------------------------------------------

def _create_feather_mask(tile_h: int, tile_w: int, overlap: int,
                         device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a feathered blending mask for tile stitching.

    Returns [1, 1, tile_h, tile_w] mask that is 1.0 in the center and fades
    to 0.0 at the edges within the overlap region.
    """
    mask = torch.ones(1, 1, tile_h, tile_w, device=device, dtype=dtype)
    if overlap <= 0:
        return mask
    # Linear ramp at each edge
    for i in range(overlap):
        alpha = (i + 1) / overlap
        mask[:, :, i, :] *= alpha          # top edge
        mask[:, :, -(i + 1), :] *= alpha   # bottom edge
        mask[:, :, :, i] *= alpha          # left edge
        mask[:, :, :, -(i + 1)] *= alpha   # right edge
    return mask


def _generate_tiles(img_h: int, img_w: int, tile_h: int, tile_w: int,
                    overlap: int, force_uniform: bool):
    """Generate tile coordinates [(y, x, h, w), ...] covering the image."""
    tiles = []
    step_h = tile_h - overlap
    step_w = tile_w - overlap

    if force_uniform:
        # Ensure tiles cover the image with uniform spacing
        n_rows = max(1, math.ceil((img_h - overlap) / step_h))
        n_cols = max(1, math.ceil((img_w - overlap) / step_w))
        for row in range(n_rows):
            for col in range(n_cols):
                y = min(row * step_h, max(0, img_h - tile_h))
                x = min(col * step_w, max(0, img_w - tile_w))
                h = min(tile_h, img_h - y)
                w = min(tile_w, img_w - x)
                tiles.append((y, x, h, w))
    else:
        y = 0
        prev_y = -1
        while y < img_h:
            if y == prev_y:
                break  # Prevent infinite loop
            prev_y = y
            x = 0
            prev_x = -1
            th = min(tile_h, img_h - y)
            while x < img_w:
                if x == prev_x:
                    break  # Prevent infinite loop
                prev_x = x
                tw = min(tile_w, img_w - x)
                tiles.append((y, x, th, tw))
                next_x = x + step_w
                if next_x >= img_w:
                    break
                if next_x + tile_w > img_w:
                    # Last tile snaps to right edge
                    x = max(0, img_w - tile_w)
                else:
                    x = next_x
            next_y = y + step_h
            if next_y >= img_h:
                break
            if next_y + tile_h > img_h:
                y = max(0, img_h - tile_h)
            else:
                y = next_y

    # Deduplicate
    seen = set()
    unique = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _generate_seam_tiles(tiles, img_h, img_w, seam_fix_width, seam_fix_mode):
    """Generate seam fix tile strips along tile boundaries."""
    seam_tiles = []
    if seam_fix_mode == "None" or seam_fix_width <= 0:
        return seam_tiles

    # Collect unique horizontal and vertical seam lines from tile boundaries
    h_seams = set()
    v_seams = set()
    for (y, x, th, tw) in tiles:
        if y > 0:
            h_seams.add(y)
        if y + th < img_h:
            h_seams.add(y + th)
        if x > 0:
            v_seams.add(x)
        if x + tw < img_w:
            v_seams.add(x + tw)

    half_w = seam_fix_width // 2

    if seam_fix_mode in ("Band", "Half Tile + Intersections"):
        # Horizontal seam bands
        for sy in sorted(h_seams):
            y_start = max(0, sy - half_w)
            y_end = min(img_h, sy + half_w)
            seam_tiles.append((y_start, 0, y_end - y_start, img_w))
        # Vertical seam bands
        for sx in sorted(v_seams):
            x_start = max(0, sx - half_w)
            x_end = min(img_w, sx + half_w)
            seam_tiles.append((0, x_start, img_h, x_end - x_start))

    if seam_fix_mode in ("Half Tile", "Half Tile + Intersections"):
        # Half-tile offset retiling
        step_h = tiles[0][2] // 2 if tiles else 256
        step_w = tiles[0][3] // 2 if tiles else 256
        for (y, x, th, tw) in tiles:
            # Offset by half tile
            hy = y + step_h
            hx = x + step_w
            if hy + th <= img_h and hx + tw <= img_w:
                seam_tiles.append((hy, hx, th, tw))

    return seam_tiles


@registry.register(
    "UltimateSDUpscale",
    return_types=("IMAGE",),
    category="image/upscaling",
    input_types={
        "required": {
            "image": ("IMAGE",),
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "upscale_model": ("UPSCALE_MODEL",),
            "seed": ("INT",),
            "steps": ("INT",),
            "cfg": ("FLOAT",),
            "sampler_name": ("STRING",),
            "scheduler": ("STRING",),
            "denoise": ("FLOAT",),
            "tile_width": ("INT",),
            "tile_height": ("INT",),
            "mask_blur": ("INT",),
            "seam_fix_mode": ("STRING",),
            "seam_fix_width": ("INT",),
            "seam_fix_denoise": ("FLOAT",),
            "force_uniform_tiles": ("BOOLEAN",),
        },
    },
)
def ultimate_sd_upscale(
    image,
    model,
    positive,
    negative,
    vae,
    upscale_model,
    seed=0,
    steps=20,
    cfg=7.0,
    sampler_name="euler",
    scheduler="normal",
    denoise=0.35,
    tile_width=512,
    tile_height=512,
    mask_blur=8,
    seam_fix_mode="None",
    seam_fix_width=64,
    seam_fix_denoise=0.35,
    force_uniform_tiles=True,
):
    """Tiled img2img upscale with optional seam fixing.

    Process:
    1. Upscale input image with upscale_model (ESRGAN etc.)
    2. Split upscaled image into overlapping tiles
    3. For each tile: VAE encode -> sample (denoise) -> VAE decode
    4. Stitch tiles back with feathered blending
    5. Optional seam fix pass on tile boundaries
    """
    from serenityflow.nodes.model_ops import image_upscale_with_model

    # Step 1: Upscale the image with the upscale model
    (upscaled,) = image_upscale_with_model(upscale_model, image)
    B, H, W, C = upscaled.shape

    # Try to get bridge sampling function
    try:
        from serenityflow.bridge.serenity_api import sample as bridge_sample
        from serenityflow.bridge.serenity_api import vae_encode, vae_decode
        from serenityflow.bridge.types import bhwc_to_bchw, bchw_to_bhwc
    except ImportError:
        # No bridge available -- return the upscaled image without img2img pass
        return (upscaled,)

    # Verify bridge is functional by testing VAE availability
    # (bridge module may exist but not be initialized)
    if vae is None:
        return (upscaled,)

    overlap = max(tile_width, tile_height) // 8  # Overlap proportional to tile size
    tiles = _generate_tiles(H, W, tile_height, tile_width, overlap, force_uniform_tiles)

    # Accumulator for blended output
    output = torch.zeros_like(upscaled)
    weight = torch.zeros(B, H, W, 1, dtype=upscaled.dtype, device=upscaled.device)

    def _process_tiles(tile_list, tile_denoise):
        """Process a list of tiles through VAE encode -> sample -> VAE decode."""
        nonlocal output, weight
        for (ty, tx, th, tw) in tile_list:
            tile_img = upscaled[:, ty:ty + th, tx:tx + tw, :].clone()

            # VAE encode (BHWC -> BCHW -> encode)
            tile_bchw = bhwc_to_bchw(tile_img)
            tile_latent = vae_encode(vae, tile_bchw)

            # Create noise mask for tile edges (blur at boundaries)
            if mask_blur > 0:
                noise_mask = torch.ones(B, th, tw, dtype=torch.float32, device=tile_img.device)
                # Blur the mask edges
                for i in range(min(mask_blur, min(th, tw) // 2)):
                    alpha = (i + 1) / mask_blur
                    noise_mask[:, i, :] *= alpha
                    noise_mask[:, -(i + 1), :] *= alpha
                    noise_mask[:, :, i] *= alpha
                    noise_mask[:, :, -(i + 1)] *= alpha
            else:
                noise_mask = None

            # Sample (denoise)
            latent_dict = wrap_latent(tile_latent, noise_mask=noise_mask)
            sampled = bridge_sample(
                model=model,
                positive=positive,
                negative=negative,
                latent=latent_dict,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=tile_denoise,
            )

            # VAE decode
            sampled_latent = unwrap_latent(sampled)
            tile_decoded_bchw = vae_decode(vae, sampled_latent)
            tile_decoded = bchw_to_bhwc(tile_decoded_bchw).clamp(0.0, 1.0)

            # Resize decoded tile if it doesn't match (VAE can round)
            if tile_decoded.shape[1] != th or tile_decoded.shape[2] != tw:
                tile_decoded = F.interpolate(
                    bhwc_to_bchw(tile_decoded), size=(th, tw),
                    mode="bilinear", align_corners=False,
                )
                tile_decoded = bchw_to_bhwc(tile_decoded)

            # Feathered blending mask
            feather = _create_feather_mask(th, tw, overlap, tile_img.device, tile_img.dtype)
            feather_bhwc = feather.permute(0, 2, 3, 1)  # [1, th, tw, 1]

            output[:, ty:ty + th, tx:tx + tw, :] += tile_decoded * feather_bhwc
            weight[:, ty:ty + th, tx:tx + tw, :] += feather_bhwc

    # Step 3-4: Process main tiles
    _process_tiles(tiles, denoise)

    # Step 5: Optional seam fix pass
    if seam_fix_mode != "None":
        seam_tiles = _generate_seam_tiles(tiles, H, W, seam_fix_width, seam_fix_mode)
        if seam_tiles:
            _process_tiles(seam_tiles, seam_fix_denoise)

    # Normalize by accumulated weights
    result = output / weight.clamp(min=1e-6)
    result = result.clamp(0.0, 1.0)

    return (result,)
