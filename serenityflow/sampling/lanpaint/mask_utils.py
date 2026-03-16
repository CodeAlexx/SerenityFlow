"""Mask utilities for LanPaint.

Handles mask reshaping to latent dimensions, binary enforcement,
and video mask support (5D tensors).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reshape_mask(
    mask: torch.Tensor,
    target_shape: tuple[int, ...],
    video: bool = False,
) -> torch.Tensor:
    """Reshape mask to match target latent shape.

    Args:
        mask: input mask, any of: (H,W), (B,H,W), (B,1,H,W), (B,1,F,H,W)
        target_shape: target latent shape (B,C,H,W) or (B,C,F,H,W) for video
        video: True for video inpainting (5D target)

    Returns:
        Mask matching target_shape dimensions.
    """
    # Normalize to at least 4D
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)

    # Handle 5D video targets
    if len(target_shape) == 5 and mask.ndim == 4:
        mask = mask.unsqueeze(2)  # (B, C, 1, H, W)

    if video and len(target_shape) == 5:
        target_frames = target_shape[2]
        target_h, target_w = target_shape[-2:]

        mask = F.interpolate(
            mask,
            size=(target_frames, target_h, target_w),
            mode="nearest-exact",
        )

        # Expand channels
        if mask.shape[1] < target_shape[1]:
            mask = mask.repeat(1, target_shape[1], 1, 1, 1)[:, :target_shape[1]]
    else:
        # 2D image case
        mask = F.interpolate(mask, size=target_shape[2:], mode="nearest-exact")
        dims = len(target_shape) - 2
        if mask.shape[1] < target_shape[1]:
            mask = mask.repeat((1, target_shape[1]) + (1,) * dims)[:, :target_shape[1]]

    # Expand batch
    if mask.shape[0] < target_shape[0]:
        repeats = [1] * mask.ndim
        repeats[0] = -(-target_shape[0] // mask.shape[0])  # ceil division
        mask = mask.repeat(*repeats)[:target_shape[0]]

    return mask


def prepare_mask(
    noise_mask: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    video: bool = False,
) -> torch.Tensor:
    """Reshape mask and move to device."""
    return reshape_mask(noise_mask, shape, video=video).to(device)


def binarize_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Enforce binary 0/1 mask. LanPaint requires hard masks."""
    return (mask > threshold).float()


__all__ = ["reshape_mask", "prepare_mask", "binarize_mask"]
