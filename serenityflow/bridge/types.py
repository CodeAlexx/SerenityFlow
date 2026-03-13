"""Type conversions between ComfyUI and Serenity tensor conventions.

ComfyUI conventions:
- Images: [B, H, W, C] float32, range [0, 1]
- Latents: {"samples": tensor, "noise_mask": optional_tensor}
- Conditioning: list[dict] with cross_attn, pooled_output keys

Serenity conventions:
- Images: [B, C, H, W] (standard PyTorch)
- Latents: raw tensors
- Conditioning: Conditioning dataclass
"""
from __future__ import annotations

import torch


def bhwc_to_bchw(image: torch.Tensor) -> torch.Tensor:
    """Convert [B, H, W, C] to [B, C, H, W]."""
    if image.ndim == 4 and image.shape[3] in (1, 3, 4):
        return image.permute(0, 3, 1, 2).contiguous()
    return image


def bchw_to_bhwc(image: torch.Tensor) -> torch.Tensor:
    """Convert [B, C, H, W] to [B, H, W, C]."""
    if image.ndim == 4 and image.shape[1] in (1, 3, 4):
        return image.permute(0, 2, 3, 1).contiguous()
    return image


def unwrap_latent(latent_dict) -> torch.Tensor:
    """Extract tensor from ComfyUI latent dict."""
    if isinstance(latent_dict, dict):
        return latent_dict["samples"]
    return latent_dict


def wrap_latent(tensor: torch.Tensor, noise_mask=None) -> dict:
    """Wrap tensor into ComfyUI latent dict."""
    result = {"samples": tensor}
    if noise_mask is not None:
        result["noise_mask"] = noise_mask
    return result


def find_cross_attn_key(cond_dict: dict) -> str | None:
    """Find the cross-attention key in a conditioning dict."""
    for key in ("cross_attn", "c_crossattn", "cond"):
        if key in cond_dict:
            return key
    return None


__all__ = [
    "bhwc_to_bchw",
    "bchw_to_bhwc",
    "unwrap_latent",
    "wrap_latent",
    "find_cross_attn_key",
]
