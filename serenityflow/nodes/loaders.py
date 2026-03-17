"""Model loader nodes -- checkpoint, UNET, CLIP, VAE, ControlNet."""
from __future__ import annotations

import logging
from pathlib import Path

from serenityflow.nodes.registry import registry

log = logging.getLogger(__name__)


def _list_models(folder: str) -> list[str]:
    """List available model files for a folder category."""
    try:
        import folder_paths
        return folder_paths.get_filename_list(folder)
    except (ImportError, Exception):
        return []


def _resolve_clip_path(clip_name: str, clip_type: str) -> str:
    """Resolve clip assets, allowing repo/path passthrough for newer families."""
    from serenityflow.bridge.model_paths import get_model_paths

    try:
        return get_model_paths().find(clip_name, "clip")
    except FileNotFoundError:
        candidate = Path(clip_name).expanduser()
        if (
            "/" in clip_name
            or candidate.exists()
            or clip_type in {"flux2", "klein", "qwen", "zimage"}
        ):
            return str(candidate) if candidate.exists() else clip_name
        raise


def _resolve_model_path(model_name: str, folder: str) -> str:
    """Resolve a model asset, allowing explicit filesystem paths."""
    from serenityflow.bridge.model_paths import get_model_paths

    try:
        return get_model_paths().find(model_name, folder)
    except FileNotFoundError:
        candidate = Path(model_name).expanduser()
        if candidate.exists() or "/" in model_name:
            return str(candidate) if candidate.exists() else model_name
        raise


@registry.register(
    "CheckpointLoaderSimple",
    return_types=("MODEL", "CLIP", "VAE"),
    return_names=("MODEL", "CLIP", "VAE"),
    category="loaders",
    input_types=lambda: {"required": {"ckpt_name": (_list_models("checkpoints") or ["(no models found)"],)}},
)
def checkpoint_loader_simple(ckpt_name):
    from serenityflow.bridge.serenity_api import load_checkpoint
    from serenityflow.bridge.model_paths import get_model_paths

    path = get_model_paths().find(ckpt_name, "checkpoints")
    model, clip, vae = load_checkpoint(path)
    return (model, clip, vae)


@registry.register(
    "UNETLoader",
    return_types=("MODEL",),
    category="loaders",
    input_types=lambda: {"required": {
        "unet_name": (_list_models("diffusion_models") or _list_models("unet") or ["(no models found)"],),
        "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
    }},
)
def unet_loader(unet_name, weight_dtype="default"):
    from serenityflow.bridge.serenity_api import load_diffusion_model

    path = _resolve_model_path(unet_name, "diffusion_models")
    model = load_diffusion_model(path, dtype=weight_dtype)
    return (model,)


@registry.register(
    "CLIPLoader",
    return_types=("CLIP",),
    category="loaders",
    input_types=lambda: {"required": {
        "clip_name": (_list_models("clip") or _list_models("text_encoders") or ["(no models found)"],),
        "type": ([
            "stable_diffusion", "stable_cascade", "sd3", "stable_audio",
            "mochi", "ltxv", "pixart", "flux", "flux2", "klein", "wan", "qwen",
            "lumina", "lumina2", "zimage",
        ],),
    }, "optional": {
        "device": (["default", "cpu", "cuda"],),
    }},
)
def clip_loader(clip_name, type="stable_diffusion", device="default"):
    from serenityflow.bridge.serenity_api import load_clip

    path = _resolve_clip_path(clip_name, type)
    clip = load_clip(path, clip_type=type)
    return (clip,)


@registry.register(
    "DualCLIPLoader",
    return_types=("CLIP",),
    category="loaders",
    input_types=lambda: {"required": {
        "clip_name1": (_list_models("clip") or _list_models("text_encoders") or ["(no models found)"],),
        "clip_name2": (_list_models("clip") or _list_models("text_encoders") or ["(no models found)"],),
        "type": (["sdxl", "sd3", "flux", "hunyuan_video"],),
    }, "optional": {
        "device": (["default", "cpu", "cuda"],),
    }},
)
def dual_clip_loader(clip_name1, clip_name2, type="flux", device="default"):
    from serenityflow.bridge.serenity_api import load_dual_clip
    from serenityflow.bridge.model_paths import get_model_paths

    paths = get_model_paths()
    path1 = paths.find(clip_name1, "clip")
    path2 = paths.find(clip_name2, "clip")
    clip = load_dual_clip(path1, path2, clip_type=type)
    return (clip,)


@registry.register(
    "VAELoader",
    return_types=("VAE",),
    category="loaders",
    input_types=lambda: {"required": {"vae_name": (_list_models("vae") or ["(no models found)"],)}},
)
def vae_loader(vae_name):
    from serenityflow.bridge.serenity_api import load_vae
    from serenityflow.bridge.model_paths import get_model_paths

    path = get_model_paths().find(vae_name, "vae")
    vae = load_vae(path)
    return (vae,)


@registry.register(
    "ControlNetLoader",
    return_types=("CONTROL_NET",),
    category="loaders",
    input_types=lambda: {"required": {"control_net_name": (_list_models("controlnet") or ["(no models found)"],)}},
)
def controlnet_loader(control_net_name):
    from serenityflow.bridge.serenity_api import load_controlnet
    from serenityflow.bridge.model_paths import get_model_paths

    path = get_model_paths().find(control_net_name, "controlnet")
    cn = load_controlnet(path)
    return (cn,)


@registry.register(
    "ImageOnlyCheckpointLoader",
    return_types=("MODEL", "CLIP_VISION", "VAE"),
    return_names=("MODEL", "CLIP_VISION", "VAE"),
    category="loaders",
    input_types={"required": {"ckpt_name": ("STRING",)}},
)
def image_only_checkpoint_loader(ckpt_name):
    raise NotImplementedError("ImageOnlyCheckpointLoader")


@registry.register(
    "QuadrupleCLIPLoader",
    return_types=("CLIP",),
    category="loaders",
    input_types={"required": {
        "clip_name1": ("STRING",), "clip_name2": ("STRING",),
        "clip_name3": ("STRING",), "clip_name4": ("STRING",),
    }},
)
def quadruple_clip_loader(clip_name1, clip_name2, clip_name3, clip_name4):
    raise NotImplementedError("QuadrupleCLIPLoader")


@registry.register(
    "ModelPatchLoader",
    return_types=("MODEL",),
    category="loaders",
    input_types={"required": {
        "model": ("MODEL",), "patch_name": ("STRING",),
    }},
)
def model_patch_loader(model, patch_name):
    raise NotImplementedError("ModelPatchLoader")


@registry.register(
    "VAEEncodeForInpaint",
    return_types=("LATENT",),
    category="latent/inpaint",
    input_types={"required": {
        "pixels": ("IMAGE",),
        "vae": ("VAE",),
        "mask": ("MASK",),
        "grow_mask_by": ("INT",),
    }},
)
def vae_encode_for_inpaint(pixels, vae, mask, grow_mask_by=6):
    raise NotImplementedError("VAEEncodeForInpaint requires bridge inpaint encoding")


@registry.register(
    "LatentUpscaleModelLoader",
    return_types=("LATENT_UPSCALE_MODEL",),
    category="loaders",
    input_types={"required": {"model_name": ("STRING",)}},
)
def latent_upscale_model_loader(model_name):
    raise NotImplementedError("LatentUpscaleModelLoader requires bridge.load_latent_upscale_model()")
