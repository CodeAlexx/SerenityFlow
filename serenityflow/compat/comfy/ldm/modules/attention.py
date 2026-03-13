"""Compatibility shim for comfy.ldm.modules.attention.

Attention implementations used by custom nodes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if not skip_reshape:
        b, _, dim_head = q.shape[0], heads, q.shape[-1] // heads
        q = q.view(b, -1, heads, dim_head).transpose(1, 2)
        k = k.view(b, -1, heads, dim_head).transpose(1, 2)
        v = v.view(b, -1, heads, dim_head).transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    if not skip_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    return attention_basic(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape)


def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    return attention_basic(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape)


def attention_sub_quad(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    return attention_basic(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape)


def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    return attention_basic(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape)


optimized_attention = attention_basic
optimized_attention_masked = attention_basic


def wrap_attn(fn):
    """Wrapper for attention function that custom nodes can intercept."""
    return fn


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64,
                 dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        if operations is None:
            linear_cls = nn.Linear
        else:
            linear_cls = getattr(operations, "Linear", nn.Linear)

        self.heads = heads
        self.dim_head = dim_head
        self.to_q = linear_cls(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = linear_cls(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = linear_cls(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_out = nn.Sequential(
            linear_cls(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, value=None, mask=None, transformer_options=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(value if value is not None else context)
        out = optimized_attention(q, k, v, self.heads, mask=mask)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0,
                 dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if operations is None:
            linear_cls = nn.Linear
        else:
            linear_cls = getattr(operations, "Linear", nn.Linear)

        act = nn.GELU()
        self.net = nn.Sequential(
            linear_cls(dim, inner_dim, dtype=dtype, device=device),
            act,
            nn.Dropout(dropout),
            linear_cls(inner_dim, dim_out, dtype=dtype, device=device),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None,
                 gated_ff=True, checkpoint=True, ff_in=False, inner_dim=None,
                 disable_self_attn=False, disable_temporal_crossattention=False,
                 switch_temporal_ca_to_sa=False, dtype=None, device=None,
                 operations=None):
        super().__init__()
        self.ff_in = ff_in
        self.is_res = inner_dim == dim
        self.disable_self_attn = disable_self_attn

        inner_dim = default(inner_dim, dim)

        self.attn1 = CrossAttention(
            query_dim=inner_dim, heads=n_heads, dim_head=d_head,
            dropout=dropout, context_dim=context_dim if disable_self_attn else None,
            dtype=dtype, device=device, operations=operations,
        )
        self.attn2 = CrossAttention(
            query_dim=inner_dim, context_dim=context_dim, heads=n_heads,
            dim_head=d_head, dropout=dropout, dtype=dtype, device=device,
            operations=operations,
        )
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout,
                              glu=gated_ff, dtype=dtype, device=device,
                              operations=operations)
        self.norm1 = nn.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm2 = nn.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = nn.LayerNorm(inner_dim, dtype=dtype, device=device)

    def forward(self, x, context=None, transformer_options=None):
        n = self.norm1(x)
        n = self.attn1(n, context=context if self.disable_self_attn else None,
                       transformer_options=transformer_options)
        x = x + n
        n = self.norm2(x)
        n = self.attn2(n, context=context, transformer_options=transformer_options)
        x = x + n
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0,
                 context_dim=None, disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = n_heads * d_head
        self.in_channels = in_channels

        if operations is None:
            linear_cls = nn.Linear
            conv_cls = nn.Conv2d
        else:
            linear_cls = getattr(operations, "Linear", nn.Linear)
            conv_cls = getattr(operations, "Conv2d", nn.Conv2d)

        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, dtype=dtype, device=device)
        self.proj_in = (
            linear_cls(in_channels, inner_dim, dtype=dtype, device=device) if use_linear
            else conv_cls(in_channels, inner_dim, kernel_size=1, stride=1, padding=0,
                          dtype=dtype, device=device)
        )

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, n_heads, d_head, dropout=dropout,
                context_dim=context_dim, disable_self_attn=disable_self_attn,
                dtype=dtype, device=device, operations=operations,
            )
            for _ in range(depth)
        ])

        self.proj_out = (
            linear_cls(inner_dim, in_channels, dtype=dtype, device=device) if use_linear
            else conv_cls(inner_dim, in_channels, kernel_size=1, stride=1, padding=0,
                          dtype=dtype, device=device)
        )
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.view(b, -1, h * w).transpose(1, 2)
        if self.use_linear:
            x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context, transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.transpose(1, 2).view(b, -1, h, w)
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
