"""ModelPatcher facade tests — clone independence, add_patches, isinstance checks."""
from __future__ import annotations

import os
import sys

compat_dir = os.path.join(os.path.dirname(__file__), "..", "serenityflow", "compat")
compat_dir = os.path.abspath(compat_dir)
if compat_dir not in sys.path:
    sys.path.insert(0, compat_dir)


def test_clone_independence():
    from comfy.model_patcher import ModelPatcher
    mp = ModelPatcher(model=None)
    mp.set_model_attn1_patch(lambda x: x)
    clone = mp.clone()
    clone.set_model_attn2_patch(lambda x: x)
    # Original should not have attn2
    assert "attn2" not in mp.model_options["transformer_options"].get("patches", {})
    # Clone should have both
    assert "attn1" in clone.model_options["transformer_options"]["patches"]
    assert "attn2" in clone.model_options["transformer_options"]["patches"]


def test_add_patches():
    from comfy.model_patcher import ModelPatcher
    mp = ModelPatcher(model=None)
    keys = mp.add_patches({"layer.weight": ("lora_up", "lora_down")}, strength_patch=0.8)
    assert "layer.weight" in keys
    assert len(mp.patches["layer.weight"]) == 1
    assert mp.patches["layer.weight"][0][0] == 0.8  # strength


def test_isinstance_check():
    """AnimateDiff does isinstance(model, ModelPatcher)."""
    from comfy.model_patcher import ModelPatcher
    mp = ModelPatcher(model=None)
    assert isinstance(mp, ModelPatcher)
    clone = mp.clone()
    assert isinstance(clone, ModelPatcher)


def test_is_clone():
    from comfy.model_patcher import ModelPatcher
    import torch
    model = torch.nn.Linear(10, 10)
    mp = ModelPatcher(model=model)
    clone = mp.clone()
    assert mp.is_clone(clone)
    assert clone.is_clone(mp)

    other = ModelPatcher(model=torch.nn.Linear(10, 10))
    assert not mp.is_clone(other)


def test_clone_has_same_weights():
    from comfy.model_patcher import ModelPatcher
    mp = ModelPatcher(model=None)
    clone = mp.clone()
    assert mp.clone_has_same_weights(clone)

    # After adding patches, UUID changes
    clone.add_patches({"k": "v"})
    assert not mp.clone_has_same_weights(clone)


def test_model_options_deep_copy():
    """model_options must be deep-copied on clone."""
    from comfy.model_patcher import ModelPatcher
    mp = ModelPatcher(model=None)
    mp.set_model_sampler_cfg_function(lambda x: x)
    clone = mp.clone()
    clone.set_model_unet_function_wrapper(lambda x: x)
    assert "model_function_wrapper" not in mp.model_options
    assert "model_function_wrapper" in clone.model_options


def test_set_model_patch_replace():
    from comfy.model_patcher import ModelPatcher
    mp = ModelPatcher(model=None)
    mp.set_model_attn1_replace(lambda x: x, "input_block", 3)
    mp.set_model_attn1_replace(lambda x: x, "output_block", 5, transformer_index=0)

    pr = mp.model_options["transformer_options"]["patches_replace"]["attn1"]
    assert ("input_block", 3) in pr
    assert ("output_block", 5, 0) in pr


def test_callbacks_and_wrappers():
    from comfy.model_patcher import ModelPatcher
    mp = ModelPatcher(model=None)

    cb1 = lambda: "cb1"
    cb2 = lambda: "cb2"
    mp.add_callback("on_load", cb1)
    mp.add_callback_with_key("on_load", "key1", cb2)

    all_cbs = mp.get_all_callbacks("on_load")
    assert cb1 in all_cbs
    assert cb2 in all_cbs

    w1 = lambda: "w1"
    mp.add_wrapper("outer_sample", w1)
    all_ws = mp.get_all_wrappers("outer_sample")
    assert w1 in all_ws


def test_module_level_helpers():
    from comfy.model_patcher import (
        set_model_options_patch_replace,
        set_model_options_post_cfg_function,
        set_model_options_pre_cfg_function,
    )

    opts = {}
    set_model_options_patch_replace(opts, lambda x: x, "attn1", "block", 0)
    assert "transformer_options" in opts
    assert "attn1" in opts["transformer_options"]["patches_replace"]

    set_model_options_post_cfg_function(opts, lambda x: x)
    assert len(opts["sampler_post_cfg_function"]) == 1

    set_model_options_pre_cfg_function(opts, lambda x: x)
    assert len(opts["sampler_pre_cfg_function"]) == 1


def test_lifecycle_methods_dont_crash():
    from comfy.model_patcher import ModelPatcher
    mp = ModelPatcher(model=None)
    mp.patch_model()
    mp.unpatch_model()
    mp.model_patches_to("cpu")
    mp.detach()
    mp.cleanup()
    mp.model_patches_models()
    mp.apply_hooks(None)
    mp.unpatch_hooks()
    assert mp.loaded_size() == 0
    assert mp.lowvram_patch_counter() == 0
    assert mp.is_dynamic() is False
