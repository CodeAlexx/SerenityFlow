"""Compatibility shim for comfy.float.

Float precision utilities.
"""
from __future__ import annotations

import torch


def stochastic_rounding(value, dtype, seed=0):
    if isinstance(value, torch.Tensor):
        return value.to(dtype)
    return value
