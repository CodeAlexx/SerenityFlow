"""Compatibility shim for comfy.ldm.modules.diffusionmodules.openaimodel.

UNet block definitions. AnimateDiff patches these.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from comfy.ldm.modules.attention import SpatialTransformer


class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        return x


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, transformer_options=None,
                output_shape=None, time_context=None, num_video_frames=None,
                image_only_indicator=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, transformer_options=transformer_options)
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

    def forward(self, x):
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

    def forward(self, x, output_shape=None):
        return x


class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 dims=2, use_checkpoint=False, use_scale_shift_norm=False,
                 up=False, down=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels

    def forward(self, x, emb):
        return x


class UNetModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.input_blocks = nn.ModuleList()
        self.middle_block = None
        self.output_blocks = nn.ModuleList()
        self.out = None

    def forward(self, x, timesteps=None, context=None, y=None,
                control=None, transformer_options=None, **kwargs):
        return x


def forward_timestep_embed(ts, x, emb, context=None, transformer_options=None,
                           output_shape=None, time_context=None,
                           num_video_frames=None, image_only_indicator=None):
    for layer in ts:
        if isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context, transformer_options=transformer_options)
        else:
            x = layer(x)
    return x
