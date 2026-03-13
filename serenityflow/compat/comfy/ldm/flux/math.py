"""Compatibility shim for comfy.ldm.flux.math.

Rotary position embeddings for Flux models.
"""
from __future__ import annotations

import torch


def apply_rope(xq, xk, freqs_cis):
    """Apply rotary position embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting
    shape = [d if i == xq_.ndim - 2 or i == xq_.ndim - 1 else 1
             for i, d in enumerate(freqs_cis.shape)]
    if len(shape) != len(xq_.shape):
        ndim = xq_.ndim
        freqs_cis = freqs_cis.view(*([1] * (ndim - freqs_cis.ndim) + list(freqs_cis.shape)))

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(xq.ndim - 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(xk.ndim - 1)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def rope(pos, dim, theta):
    """Generate rotary position embedding frequencies."""
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.float(), omega.float())
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.view(*out.shape[:-1], 2, 2)
    return out.float()


def attention(q, k, v, pe, mask=None):
    """Attention with rotary position embeddings."""
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x.transpose(1, 2).contiguous()
    return x
