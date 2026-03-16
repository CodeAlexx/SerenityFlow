"""ControlNet nodes."""
from __future__ import annotations

from serenityflow.nodes.registry import registry


@registry.register(
    "ControlNetApply",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "control_net": ("CONTROL_NET",),
        "image": ("IMAGE",),
        "strength": ("FLOAT",),
    }},
)
def controlnet_apply(conditioning, control_net, image, strength=1.0):
    out = []
    for c in conditioning:
        n = dict(c)
        prev_hints = list(n.get("control_hints", []))
        prev_hints.append({
            "control_net": control_net,
            "hint_image": image,
            "strength": strength,
        })
        n["control_hints"] = prev_hints
        out.append(n)
    return (out,)


@registry.register(
    "ControlNetApplyAdvanced",
    return_types=("CONDITIONING", "CONDITIONING"),
    category="conditioning",
    input_types={"required": {
        "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
        "control_net": ("CONTROL_NET",), "image": ("IMAGE",),
        "strength": ("FLOAT",),
        "start_percent": ("FLOAT",), "end_percent": ("FLOAT",),
    }},
)
def controlnet_apply_advanced(positive, negative, control_net, image,
                              strength=1.0, start_percent=0.0, end_percent=1.0):
    from serenityflow.bridge.serenity_api import apply_controlnet

    pos_out, neg_out = apply_controlnet(
        positive, negative, control_net, image,
        strength=strength, start_percent=start_percent, end_percent=end_percent,
    )
    return (pos_out, neg_out)
