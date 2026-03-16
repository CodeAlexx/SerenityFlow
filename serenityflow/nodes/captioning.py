"""Captioning nodes -- JoyCaption and WD14 tagger for image-to-text."""
from __future__ import annotations

from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# JoyCaption
# ---------------------------------------------------------------------------

@registry.register(
    "JoyCaptionModelLoader",
    return_types=("JOYCAPTION_MODEL",),
    category="captioning",
    input_types={"required": {"model_name": ("STRING",)}},
)
def joycaption_model_loader(model_name):
    """Load a JoyCaption model by name.

    Returns a handle dict that downstream nodes use to run inference.
    The bridge layer handles actual weight loading at execution time.
    """
    try:
        from serenityflow.bridge.serenity_api import load_joycaption
        model = load_joycaption(model_name)
    except (ImportError, AttributeError):
        # Bridge function not yet implemented -- return a lazy handle
        model = {"_type": "joycaption", "model_name": model_name}
    return (model,)


@registry.register(
    "JoyCaptionAdvanced",
    return_types=("STRING",),
    category="captioning",
    input_types={
        "required": {
            "joycaption_model": ("JOYCAPTION_MODEL",),
            "image": ("IMAGE",),
        },
        "optional": {
            "mode": ("STRING",),
            "length": ("STRING",),
            "focus_on": ("STRING",),
        },
    },
)
def joycaption_advanced(joycaption_model, image, mode="descriptive",
                        length="medium", focus_on="general"):
    """Generate a caption from an image using JoyCaption.

    Supports multiple captioning modes, length controls, and focus areas.
    """
    # If the model is a real callable / nn.Module, run it
    if hasattr(joycaption_model, "caption") and callable(joycaption_model.caption):
        caption = joycaption_model.caption(
            image, mode=mode, length=length, focus_on=focus_on,
        )
        return (caption,)

    # If it's a bridge handle dict, delegate
    if isinstance(joycaption_model, dict) and joycaption_model.get("_type") == "joycaption":
        try:
            from serenityflow.bridge.serenity_api import run_joycaption
            caption = run_joycaption(
                joycaption_model, image,
                mode=mode, length=length, focus_on=focus_on,
            )
            return (caption,)
        except (ImportError, AttributeError):
            pass

    raise NotImplementedError(
        "JoyCaptionAdvanced: model does not expose .caption() and "
        "bridge.run_joycaption() is not available"
    )


# ---------------------------------------------------------------------------
# WD14 Tagger
# ---------------------------------------------------------------------------

@registry.register(
    "WD14ModelLoader",
    return_types=("WD14_MODEL",),
    category="captioning",
    input_types={"required": {"model_name": ("STRING",)}},
)
def wd14_model_loader(model_name):
    """Load a WD14 tagger model by name."""
    try:
        from serenityflow.bridge.serenity_api import load_wd14
        model = load_wd14(model_name)
    except (ImportError, AttributeError):
        model = {"_type": "wd14", "model_name": model_name}
    return (model,)


@registry.register(
    "WD14Tag",
    return_types=("STRING",),
    category="captioning",
    input_types={
        "required": {
            "wd14_model": ("WD14_MODEL",),
            "image": ("IMAGE",),
        },
        "optional": {
            "threshold": ("FLOAT",),
            "character_threshold": ("FLOAT",),
        },
    },
)
def wd14_tag(wd14_model, image, threshold=0.35, character_threshold=0.85):
    """Generate booru-style tags from an image using WD14 tagger.

    Returns comma-separated tags above the confidence threshold.
    """
    # Real model with .tag() method
    if hasattr(wd14_model, "tag") and callable(wd14_model.tag):
        tags = wd14_model.tag(
            image, threshold=threshold, character_threshold=character_threshold,
        )
        if isinstance(tags, list):
            tags = ", ".join(tags)
        return (tags,)

    # Bridge handle
    if isinstance(wd14_model, dict) and wd14_model.get("_type") == "wd14":
        try:
            from serenityflow.bridge.serenity_api import run_wd14
            tags = run_wd14(
                wd14_model, image,
                threshold=threshold, character_threshold=character_threshold,
            )
            if isinstance(tags, list):
                tags = ", ".join(tags)
            return (tags,)
        except (ImportError, AttributeError):
            pass

    raise NotImplementedError(
        "WD14Tag: model does not expose .tag() and "
        "bridge.run_wd14() is not available"
    )
