"""Compatibility shim for comfy.diffusers_convert.

State dict conversion between diffusers and comfy formats.
"""
from __future__ import annotations


def convert_unet_state_dict(unet_state_dict):
    return unet_state_dict


def convert_vae_state_dict(vae_state_dict):
    return vae_state_dict


def convert_text_enc_state_dict(text_enc_state_dict):
    return text_enc_state_dict


def convert_text_enc_state_dict_v20(text_enc_state_dict):
    return text_enc_state_dict
