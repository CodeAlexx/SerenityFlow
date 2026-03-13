"""Compatibility shim for comfy.ldm.util."""
from __future__ import annotations

import torch


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
