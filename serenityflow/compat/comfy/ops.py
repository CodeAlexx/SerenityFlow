"""Compatibility shim for comfy.ops.

Provides CastWeightBiasOp and manual_cast wrapper classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import comfy.model_management


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    weight = s.weight
    bias = s.bias if hasattr(s, "bias") else None
    non_blocking = comfy.model_management.device_supports_non_blocking(device) if device else False

    if device is not None or dtype is not None:
        weight = weight.to(device=device, dtype=dtype, non_blocking=non_blocking)
    if bias is not None and (device is not None or bias_dtype is not None or dtype is not None):
        bd = bias_dtype or dtype
        bias = bias.to(device=device, dtype=bd, non_blocking=non_blocking)
    return weight, bias


class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = None
    bias_function = None


class disable_weight_init:
    """nn.Module subclasses that skip default initialization."""

    class Linear(nn.Linear):
        comfy_cast_weights = False
        weight_function = None
        bias_function = None

        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Conv1d(nn.Conv1d):
        comfy_cast_weights = False
        weight_function = None
        bias_function = None

        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Conv2d(nn.Conv2d):
        comfy_cast_weights = False
        weight_function = None
        bias_function = None

        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Conv3d(nn.Conv3d):
        comfy_cast_weights = False
        weight_function = None
        bias_function = None

        def reset_parameters(self):
            return None

    class GroupNorm(nn.GroupNorm):
        comfy_cast_weights = False
        weight_function = None
        bias_function = None

        def reset_parameters(self):
            return None

    class LayerNorm(nn.LayerNorm):
        comfy_cast_weights = False
        weight_function = None
        bias_function = None

        def reset_parameters(self):
            return None

    class Embedding(nn.Embedding):
        comfy_cast_weights = False
        weight_function = None
        bias_function = None

        def reset_parameters(self):
            self.weight.data.normal_()


class manual_cast(disable_weight_init):
    """Same as disable_weight_init but with comfy_cast_weights = True."""

    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True
