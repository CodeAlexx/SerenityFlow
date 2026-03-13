"""Compatibility shim for nodes module.

Exposes NODE_CLASS_MAPPINGS and built-in node class stubs.
Custom nodes do `from nodes import NODE_CLASS_MAPPINGS` or import specific node classes.
"""
from __future__ import annotations

MAX_RESOLUTION = 16384

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}


def init_builtin_extra_nodes():
    """Called during startup to register extra built-in nodes."""
    pass


def init_external_custom_nodes():
    """Called during startup to load custom nodes."""
    pass


# Stub classes for common built-in nodes that custom nodes may import directly

class KSampler:
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 8.0}),
                "sampler_name": (["euler"],),
                "scheduler": (["normal"],),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0}),
            }
        }


class CheckpointLoaderSimple:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"


class CLIPTextEncode:
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"


class VAEDecode:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"


class VAEEncode:
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "latent"


class EmptyLatentImage:
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent"


class LoraLoader:
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"


class SaveImage:
    RETURN_TYPES = ()
    FUNCTION = "save_images"
    CATEGORY = "image"
    OUTPUT_NODE = True


class PreviewImage(SaveImage):
    pass


class LoadImage:
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image"


class ControlNetLoader:
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders"


class ControlNetApply:
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/controlnet"


class ControlNetApplyAdvanced:
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/controlnet"


class CLIPVisionLoader:
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip_vision"
    CATEGORY = "loaders"


class CLIPVisionEncode:
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/style_model"


class ConditioningCombine:
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "conditioning"


class ConditioningSetArea:
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"
    CATEGORY = "conditioning"


class ConditioningSetMask:
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"
    CATEGORY = "conditioning"


# Register all stubs
_BUILTIN_NODES = {
    "KSampler": KSampler,
    "CheckpointLoaderSimple": CheckpointLoaderSimple,
    "CLIPTextEncode": CLIPTextEncode,
    "VAEDecode": VAEDecode,
    "VAEEncode": VAEEncode,
    "EmptyLatentImage": EmptyLatentImage,
    "LoraLoader": LoraLoader,
    "SaveImage": SaveImage,
    "PreviewImage": PreviewImage,
    "LoadImage": LoadImage,
    "ControlNetLoader": ControlNetLoader,
    "ControlNetApply": ControlNetApply,
    "ControlNetApplyAdvanced": ControlNetApplyAdvanced,
    "CLIPVisionLoader": CLIPVisionLoader,
    "CLIPVisionEncode": CLIPVisionEncode,
    "ConditioningCombine": ConditioningCombine,
    "ConditioningSetArea": ConditioningSetArea,
    "ConditioningSetMask": ConditioningSetMask,
}
NODE_CLASS_MAPPINGS.update(_BUILTIN_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k in _BUILTIN_NODES})
