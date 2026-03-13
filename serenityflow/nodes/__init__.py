"""Import all node modules to trigger registration."""

from serenityflow.nodes import (  # noqa: F401
    loaders,
    sampling,
    sampling_custom,
    conditioning,
    latent,
    image_io,
    image_ops,
    post_processing,
    mask,
    lora,
    controlnet,
    video,
    utility,
    model_ops,
    model_specific,
    conditioning_extra,
    audio,
)
