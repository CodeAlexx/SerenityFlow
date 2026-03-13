"""Compatibility shim for comfy.clip_vision."""
from __future__ import annotations

import torch

import comfy.utils


class Output:
    def __init__(self):
        self.last_hidden_state = None
        self.image_embeds = None
        self.penultimate_hidden_states = None


class ClipVisionModel:
    def __init__(self, config_dict=None, device="cpu", dtype=None):
        self.model = None
        self.device = device
        self.dtype = dtype or torch.float32
        self.patcher = None

    def encode_image(self, image):
        out = Output()
        out.last_hidden_state = torch.zeros(1, 257, 1024)
        out.image_embeds = torch.zeros(1, 768)
        out.penultimate_hidden_states = torch.zeros(1, 257, 1024)
        return out

    def load_sd(self, sd):
        pass


def load(ckpt_path, model_options=None):
    return ClipVisionModel()


def clip_preprocess(image, size=224):
    if image.ndim == 4:
        image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(size, size), mode="bilinear")
    return image
