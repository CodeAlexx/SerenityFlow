"""Compatibility shim for comfy.cli_args.

Provides args namespace that custom nodes may check.
"""
from __future__ import annotations

import argparse


args = argparse.Namespace(
    listen="127.0.0.1",
    port=8188,
    cuda_device=None,
    cuda_malloc=False,
    disable_cuda_malloc=False,
    dont_upcast_attention=False,
    force_fp32=False,
    force_fp16=False,
    bf16_unet=False,
    fp16_unet=False,
    fp8_e4m3fn_unet=False,
    fp8_e5m2_unet=False,
    fp16_vae=False,
    fp32_vae=False,
    bf16_vae=False,
    cpu_vae=False,
    disable_xformers=False,
    force_upcast_attention=False,
    use_split_cross_attention=False,
    use_quad_cross_attention=False,
    use_pytorch_cross_attention=True,
    disable_ipex_optimize=False,
    lowvram=False,
    novram=False,
    highvram=False,
    normalvram=False,
    cpu=False,
    preview_method="auto",
    output_directory="output",
    temp_directory=None,
    input_directory="input",
    auto_launch=False,
    disable_auto_launch=False,
    extra_model_paths_config=None,
    disable_metadata=False,
    windows_standalone_build=False,
    multi_user=False,
    verbose=False,
)


def enable_args_parsing():
    pass
