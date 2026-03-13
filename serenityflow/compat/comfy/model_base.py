"""Compatibility shim for comfy.model_base.

Architecture type classes. AnimateDiff type-checks against these.
"""
from __future__ import annotations

from enum import Enum

import torch


class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5
    FLOW = 6
    V_PREDICTION_CONTINUOUS = 7
    FLUX = 8


class BaseModel(torch.nn.Module):
    def __init__(self, model_config=None, model_type=ModelType.EPS, device=None,
                 unet_model=None):
        super().__init__()
        self.model_type = model_type
        self.model_config = model_config
        self.diffusion_model = unet_model
        self.latent_format = None
        self.model_sampling = None
        self.adm_channels = 0
        self.inpaint_model = False
        self.concat_keys = ()
        self.memory_usage_factor = 1.0
        self.manual_cast_dtype = None

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                    transformer_options=None, **kwargs):
        if self.diffusion_model is not None:
            return self.diffusion_model(x, t)
        return x

    def get_dtype(self):
        if self.diffusion_model is not None:
            return next(self.diffusion_model.parameters()).dtype
        return torch.float32

    def is_adm(self):
        return self.adm_channels > 0

    def encode_adm(self, **kwargs):
        return None

    def extra_conds(self, **kwargs):
        return {}

    def model_size(self):
        if self.diffusion_model is not None:
            return sum(p.numel() * p.element_size() for p in self.diffusion_model.parameters())
        return 0


class SD15(BaseModel):
    def __init__(self, model_config=None, **kwargs):
        super().__init__(model_config=model_config, model_type=ModelType.EPS, **kwargs)


class SDXL(BaseModel):
    def __init__(self, model_config=None, **kwargs):
        super().__init__(model_config=model_config, model_type=ModelType.EPS, **kwargs)
        self.adm_channels = 2816


class SDXLRefiner(BaseModel):
    def __init__(self, model_config=None, **kwargs):
        super().__init__(model_config=model_config, model_type=ModelType.EPS, **kwargs)
        self.adm_channels = 2560


class SD3(BaseModel):
    def __init__(self, model_config=None, **kwargs):
        super().__init__(model_config=model_config, model_type=ModelType.FLOW, **kwargs)


class AuraFlow(BaseModel):
    def __init__(self, model_config=None, **kwargs):
        super().__init__(model_config=model_config, model_type=ModelType.FLOW, **kwargs)


class Flux(BaseModel):
    def __init__(self, model_config=None, **kwargs):
        super().__init__(model_config=model_config, model_type=ModelType.FLUX, **kwargs)


class SVD_img2vid(BaseModel):
    def __init__(self, model_config=None, **kwargs):
        super().__init__(model_config=model_config, model_type=ModelType.V_PREDICTION_EDM, **kwargs)
