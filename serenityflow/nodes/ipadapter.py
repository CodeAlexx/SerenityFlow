"""IP-Adapter nodes -- model loading and application."""
from __future__ import annotations

from serenityflow.nodes.registry import registry


@registry.register(
    "IPAdapterModelLoader",
    return_types=("IPADAPTER_MODEL",),
    category="loaders",
    input_types={"required": {"model_name": ("STRING",)}},
)
def ipadapter_model_loader(model_name):
    try:
        from serenityflow.bridge.serenity_api import load_ipadapter
        from serenityflow.bridge.model_paths import get_model_paths
        paths = get_model_paths()
        path = paths.find(model_name, "ipadapter")
        return (load_ipadapter(path),)
    except (ImportError, NotImplementedError, AttributeError):
        # Return a path handle for deferred loading
        return ({"model_name": model_name, "type": "ipadapter"},)


@registry.register(
    "IPAdapterApplyFull",
    return_types=("MODEL",),
    category="conditioning/ipadapter",
    input_types={
        "required": {
            "model": ("MODEL",),
            "ipadapter": ("IPADAPTER_MODEL",),
            "clip_vision": ("CLIP_VISION",),
            "image": ("IMAGE",),
            "weight": ("FLOAT",),
            "weight_type": ("STRING",),
            "start_at": ("FLOAT",),
            "end_at": ("FLOAT",),
            "unfold_batch": ("BOOLEAN",),
        },
    },
)
def ipadapter_apply_full(model, ipadapter, clip_vision, image,
                         weight=1.0, weight_type="linear",
                         start_at=0.0, end_at=1.0, unfold_batch=False):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "ipadapter": {
                "model": ipadapter,
                "clip_vision": clip_vision,
                "image": image,
                "weight": weight,
                "weight_type": weight_type,
                "start_at": start_at,
                "end_at": end_at,
                "unfold_batch": unfold_batch,
            },
        }),)
    return (model,)


@registry.register(
    "IPAdapterApplyFaceID",
    return_types=("MODEL",),
    category="conditioning/ipadapter",
    input_types={
        "required": {
            "model": ("MODEL",),
            "ipadapter": ("IPADAPTER_MODEL",),
            "clip_vision": ("CLIP_VISION",),
            "image": ("IMAGE",),
            "weight": ("FLOAT",),
            "start_at": ("FLOAT",),
            "end_at": ("FLOAT",),
        },
    },
)
def ipadapter_apply_face_id(model, ipadapter, clip_vision, image,
                            weight=1.0, start_at=0.0, end_at=1.0):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "ipadapter": {
                "model": ipadapter,
                "clip_vision": clip_vision,
                "image": image,
                "weight": weight,
                "weight_type": "linear",
                "start_at": start_at,
                "end_at": end_at,
                "face_id": True,
            },
        }),)
    return (model,)
