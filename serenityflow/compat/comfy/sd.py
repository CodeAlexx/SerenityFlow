"""Compatibility shim for comfy.sd.

Provides CLIP, VAE, load_checkpoint_guess_config, load_lora_for_models.
Delegates to bridge for actual loading.
"""
from __future__ import annotations

import torch


class CLIP:
    """CLIP wrapper for custom node compatibility."""

    def __init__(self, target=None, embedding_directory=None, tokenizer_data=None,
                 parameters=0, model_options=None):
        self.patcher = None
        self.cond_stage_model = target
        self.tokenizer = None
        self.layer_idx = None
        self.embedding_directory = embedding_directory

    def clone(self):
        import copy
        n = CLIP.__new__(CLIP)
        n.patcher = self.patcher.clone() if self.patcher else None
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        n.embedding_directory = self.embedding_directory
        return n

    def tokenize(self, text, return_word_ids=False):
        return {}

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        cond = torch.zeros(1, 77, 768)
        pooled = torch.zeros(1, 768)
        if return_dict:
            return cond, {"pooled_output": pooled}
        if return_pooled:
            return cond, pooled
        return cond

    def load_sd(self, sd):
        pass

    def get_sd(self):
        return {}

    def load_model(self):
        pass

    def set_clip_options(self, options):
        pass

    def reset_clip_options(self):
        pass


class VAE:
    """VAE wrapper for custom node compatibility."""

    def __init__(self, sd=None, device=None, config=None, dtype=None):
        self.first_stage_model = None
        self.memory_used_encode = lambda shape, dtype: 0
        self.memory_used_decode = lambda shape, dtype: 0
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.output_channels = 3
        self.process_input = lambda x: x
        self.process_output = lambda x: x

    def decode(self, samples_in):
        return torch.zeros(1, 3, samples_in.shape[2] * 8, samples_in.shape[3] * 8)

    def encode(self, pixel_samples):
        return torch.zeros(
            1, self.latent_channels,
            pixel_samples.shape[1] // 8, pixel_samples.shape[2] // 8,
        )

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        return self.decode(samples)

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        return self.encode(pixel_samples)

    def get_sd(self):
        return {}


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                  embedding_directory=None, output_model=True,
                                  model_options=None):
    """Load a checkpoint. Returns (model_patcher, clip, vae, clipvision)."""
    from comfy.model_patcher import ModelPatcher
    model = ModelPatcher(model=None) if output_model else None
    clip = CLIP() if output_clip else None
    vae = VAE() if output_vae else None
    return model, clip, vae, None


def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
    """Apply LoRA to model and clip. Returns (new_model, new_clip)."""
    new_model = model.clone() if model is not None else None
    new_clip = clip.clone() if clip is not None else None

    if new_model is not None and lora is not None:
        model_keys = {k: v for k, v in lora.items() if not k.startswith("lora_te")}
        if model_keys:
            new_model.add_patches(model_keys, strength_patch=strength_model)

    if new_clip is not None and new_clip.patcher is not None and lora is not None:
        clip_keys = {k: v for k, v in lora.items() if k.startswith("lora_te")}
        if clip_keys:
            new_clip.patcher.add_patches(clip_keys, strength_patch=strength_clip)

    return new_model, new_clip


def load_clip(ckpt_paths, embedding_directory=None, clip_type=None, model_options=None):
    return CLIP(embedding_directory=embedding_directory)


def load_state_dict_guess_config(sd, output_vae=True, output_clip=True,
                                  output_clipvision=False, embedding_directory=None,
                                  output_model=True, model_options=None):
    return load_checkpoint_guess_config(
        "", output_vae=output_vae, output_clip=output_clip,
        embedding_directory=embedding_directory, output_model=output_model,
        model_options=model_options,
    )
