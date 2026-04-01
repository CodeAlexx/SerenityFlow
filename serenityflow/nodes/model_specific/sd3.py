"""SD3-specific nodes -- triple CLIP encoder, latent."""
from __future__ import annotations

from serenityflow.nodes.registry import registry


def _list_models(folder):
    from serenityflow.nodes.loaders import _list_models as _lm
    return _lm(folder)


@registry.register(
    "TripleCLIPLoader",
    return_types=("CLIP",),
    category="advanced/loaders",
    input_types=lambda: {"required": {
        "clip_name1": (_list_models("clip") or _list_models("text_encoders") or ["(no models found)"],),
        "clip_name2": (_list_models("clip") or _list_models("text_encoders") or ["(no models found)"],),
        "clip_name3": (_list_models("clip") or _list_models("text_encoders") or ["(no models found)"],),
    }},
)
def triple_clip_loader(clip_name1, clip_name2, clip_name3):
    from serenityflow.bridge.serenity_api import load_triple_clip
    from serenityflow.bridge.model_paths import get_model_paths

    paths = get_model_paths()
    path1 = paths.find(clip_name1, "clip")
    path2 = paths.find(clip_name2, "clip")
    path3 = paths.find(clip_name3, "clip")
    clip = load_triple_clip(path1, path2, path3, clip_type="sd3")
    return (clip,)


@registry.register(
    "CLIPTextEncodeSD3",
    return_types=("CONDITIONING",),
    category="conditioning/sd3",
    input_types={"required": {
        "clip": ("CLIP",),
        "clip_l": ("STRING",),
        "clip_g": ("STRING",),
        "t5xxl": ("STRING",),
    },
    "optional": {
        "empty_padding": ("STRING",),
    }},
)
def clip_text_encode_sd3(clip, clip_l, clip_g, t5xxl, empty_padding="none"):
    from serenityflow.bridge.serenity_api import encode_text
    conditioning = encode_text(clip, t5xxl)
    out = []
    for c in conditioning:
        n = dict(c)
        n["clip_l_text"] = clip_l
        n["clip_g_text"] = clip_g
        n["t5xxl_text"] = t5xxl
        out.append(n)
    return (out,)
