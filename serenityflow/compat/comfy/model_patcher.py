"""ModelPatcher facade for ComfyUI custom node compatibility.

Custom nodes use clone(), add_patches(), set_model_attn*_patch().
Routes to ModelHandle + PatchLedger internally.
"""
from __future__ import annotations

import copy
import uuid

import torch
import comfy.model_management


class ModelPatcher:
    def __init__(self, model=None, load_device=None, offload_device=None,
                 size=0, weight_inplace_update=False):
        if model is not None and hasattr(model, "handle_id"):
            self._handle = model
        else:
            self._handle = None
        self._raw_model = model

        self.model = model
        self.load_device = load_device or comfy.model_management.get_torch_device()
        self.offload_device = offload_device or torch.device("cpu")
        self.size = size
        self.weight_inplace_update = weight_inplace_update
        self.patches = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.patches_uuid = uuid.uuid4()
        self.parent = None
        self.backup = {}
        self.attachments = {}
        self.additional_models = {}
        self.callbacks = {}
        self.wrappers = {}
        self.hook_patches = {}
        self.current_hooks = None
        self.forced_hooks = None
        self.is_clip = False

    def clone(self):
        n = ModelPatcher.__new__(ModelPatcher)
        n._handle = self._handle
        n._raw_model = self._raw_model
        n.model = self.model
        n.load_device = self.load_device
        n.offload_device = self.offload_device
        n.size = self.size
        n.weight_inplace_update = self.weight_inplace_update
        n.patches = {k: v[:] for k, v in self.patches.items()}
        n.object_patches = self.object_patches.copy()
        n.object_patches_backup = {}
        n.model_options = copy.deepcopy(self.model_options)
        n.patches_uuid = self.patches_uuid
        n.parent = self
        n.backup = self.backup
        n.attachments = {}
        n.additional_models = {}
        n.callbacks = {}
        n.wrappers = {}
        n.hook_patches = {}
        n.current_hooks = self.current_hooks
        n.forced_hooks = self.forced_hooks
        n.is_clip = self.is_clip
        return n

    def is_clone(self, other):
        if not isinstance(other, ModelPatcher):
            return False
        return self.model is other.model

    def clone_has_same_weights(self, clone):
        if not self.is_clone(clone):
            return False
        return self.patches_uuid == clone.patches_uuid

    def model_size(self):
        if self.size > 0:
            return self.size
        if hasattr(self.model, "model_size"):
            return self.model.model_size
        if self.model is not None and hasattr(self.model, "state_dict"):
            return comfy.model_management.module_size(self.model)
        return 0

    def loaded_size(self):
        return 0

    def lowvram_patch_counter(self):
        return 0

    def model_dtype(self):
        if self._handle and hasattr(self._handle, "dtype"):
            return self._handle.dtype
        return torch.float32

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for key in patches:
            self.patches.setdefault(key, []).append(
                (strength_patch, patches[key], strength_model)
            )
            p.add(key)
        self.patches_uuid = uuid.uuid4()
        return list(p)

    def get_key_patches(self, filter_prefix=None):
        return self.patches

    def memory_required(self, input_shape=None):
        return self.model_size()

    # === model_options manipulation ===

    def set_model_sampler_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_cfg_function"] = fn
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options.setdefault("sampler_post_cfg_function", []).append(fn)
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_pre_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options.setdefault("sampler_pre_cfg_function", []).append(fn)
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def set_model_denoise_mask_function(self, fn):
        self.model_options["denoise_mask_function"] = fn

    # === Attention / block patching ===

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        to.setdefault("patches", {}).setdefault(name, []).append(patch)

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        to = self.model_options["transformer_options"]
        pr = to.setdefault("patches_replace", {}).setdefault(name, {})
        key = (block_name, number) if transformer_index is None else (block_name, number, transformer_index)
        pr[key] = patch

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block")

    def set_model_emb_patch(self, patch):
        self.set_model_patch(patch, "emb")

    def set_model_forward_timestep_embed_patch(self, patch):
        self.set_model_patch(patch, "forward_timestep_embed")

    def set_model_double_block_patch(self, patch):
        self.set_model_patch(patch, "double_block")

    def set_model_rope_options(self, **kwargs):
        self.model_options["transformer_options"]["rope_options"] = kwargs

    # === Callbacks / wrappers ===

    def add_callback(self, call_type, callback):
        self.callbacks.setdefault(call_type, {}).setdefault(None, []).append(callback)

    def add_callback_with_key(self, call_type, key, callback):
        self.callbacks.setdefault(call_type, {}).setdefault(key, []).append(callback)

    def get_all_callbacks(self, call_type):
        result = []
        for cbs in self.callbacks.get(call_type, {}).values():
            result.extend(cbs)
        return result

    def add_wrapper(self, wrapper_type, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(None, []).append(wrapper)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def get_all_wrappers(self, wrapper_type):
        result = []
        for ws in self.wrappers.get(wrapper_type, {}).values():
            result.extend(ws)
        return result

    # === Lifecycle ===

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True,
                    force_patch_weights=False):
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        pass

    def model_patches_to(self, device):
        pass

    def detach(self, unpatch_all=True):
        return self.model

    def current_loaded_device(self):
        return self.load_device

    def is_dynamic(self):
        return False

    def cleanup(self):
        pass

    def model_patches_models(self):
        return []

    # === Hook support ===

    def apply_hooks(self, hooks, force_apply=False):
        self.current_hooks = hooks

    def unpatch_hooks(self):
        pass


# Module-level helper functions

def set_model_options_patch_replace(model_options, patch, name, block_name, number,
                                    transformer_index=None):
    to = model_options.setdefault("transformer_options", {})
    pr = to.setdefault("patches_replace", {}).setdefault(name, {})
    key = (block_name, number) if transformer_index is None else (block_name, number, transformer_index)
    pr[key] = patch
    return model_options


def set_model_options_post_cfg_function(model_options, fn, disable_cfg1_optimization=False):
    model_options.setdefault("sampler_post_cfg_function", []).append(fn)
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


def set_model_options_pre_cfg_function(model_options, fn, disable_cfg1_optimization=False):
    model_options.setdefault("sampler_pre_cfg_function", []).append(fn)
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options
