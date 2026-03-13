"""LoRA loader nodes."""
from __future__ import annotations

from serenityflow.nodes.registry import registry


@registry.register(
    "LoraLoader",
    return_types=("MODEL", "CLIP"),
    category="loaders",
    input_types={"required": {
        "model": ("MODEL",), "clip": ("CLIP",),
        "lora_name": ("STRING",),
        "strength_model": ("FLOAT",), "strength_clip": ("FLOAT",),
    }},
)
def lora_loader(model, clip, lora_name, strength_model, strength_clip):
    from serenityflow.bridge.serenity_api import apply_lora, apply_lora_clip
    from serenityflow.bridge.model_paths import get_model_paths

    lora_path = get_model_paths().find(lora_name, "loras")
    new_model = apply_lora(model, lora_path, strength=strength_model)
    new_clip = apply_lora_clip(clip, lora_path, strength=strength_clip)
    return (new_model, new_clip)


@registry.register(
    "LoraLoaderModelOnly",
    return_types=("MODEL",),
    category="loaders",
    input_types={"required": {
        "model": ("MODEL",), "lora_name": ("STRING",),
        "strength_model": ("FLOAT",),
    }},
)
def lora_loader_model_only(model, lora_name, strength_model):
    from serenityflow.bridge.serenity_api import apply_lora
    from serenityflow.bridge.model_paths import get_model_paths

    lora_path = get_model_paths().find(lora_name, "loras")
    new_model = apply_lora(model, lora_path, strength=strength_model)
    return (new_model,)
