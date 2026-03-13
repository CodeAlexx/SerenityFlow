"""Compatibility shim for comfy.k_diffusion.sa_solver."""
from __future__ import annotations

import torch


def sample_sa_solver(model, x, sigmas, extra_args=None, callback=None,
                     disable=None, **kwargs):
    from comfy.k_diffusion.sampling import sample_euler
    return sample_euler(model, x, sigmas, extra_args=extra_args,
                       callback=callback, disable=disable)
