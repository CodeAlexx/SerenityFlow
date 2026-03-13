"""Compatibility shim for comfy.controlnet.

Provides ControlBase and broadcast_image_to helper.
"""
from __future__ import annotations

import torch


def broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    if current_batch_size == 1:
        return tensor.repeat(target_batch_size, 1, 1, 1)
    per_batch = target_batch_size // batched_number
    return tensor.repeat(per_batch, 1, 1, 1)[:target_batch_size]


class ControlBase:
    def __init__(self, device=None):
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.timestep_percent_range = (0.0, 1.0)
        self.latent_format = None
        self.vae = None
        self.device = device
        self.previous_controlnet = None
        self.extra_args = {}
        self.extra_conds = []

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0),
                      vae=None, extra_concat=None):
        self.cond_hint_original = cond_hint
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range
        self.vae = vae
        return self

    def pre_run(self, model, percent_to_timestep_function):
        pass

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet

    def cleanup(self):
        self.cond_hint = None

    def get_models(self):
        return []

    def copy_to(self, c):
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        c.timestep_percent_range = self.timestep_percent_range
        c.latent_format = self.latent_format
        c.vae = self.vae
        c.device = self.device
        c.previous_controlnet = self.previous_controlnet
        c.extra_args = self.extra_args.copy()

    def get_control(self, x, timestep, cond, batched_number):
        return {}

    def copy(self):
        import copy
        c = copy.copy(self)
        return c
