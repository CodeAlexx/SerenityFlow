"""Tensor utilities shim for comfy.utils.

Mostly pure math/torch. File loading via safetensors.
"""
from __future__ import annotations

import json
import math
import os

import torch


# === File I/O ===


def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    if device is None:
        device = "cpu"
    ext = os.path.splitext(ckpt)[1].lower()
    if ext == ".safetensors":
        import safetensors.torch
        if return_metadata:
            with open(ckpt, "rb") as f:
                header_size = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(header_size))
                metadata = header.get("__metadata__", {})
            sd = safetensors.torch.load_file(ckpt, device=str(device))
            return sd, metadata
        return safetensors.torch.load_file(ckpt, device=str(device))
    else:
        sd = torch.load(ckpt, map_location=device, weights_only=safe_load)
        if return_metadata:
            return sd, {}
        return sd


def save_torch_file(sd, ckpt, metadata=None):
    import safetensors.torch
    safetensors.torch.save_file(sd, ckpt, metadata=metadata)


# === Tensor ops ===


def common_upscale(samples, width, height, upscale_method, crop):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:, :, y:old_height - y, x:old_width - x]
    else:
        s = samples
    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


def repeat_to_batch_size(tensor, batch_size, dim=0):
    if tensor.shape[dim] >= batch_size:
        return tensor.narrow(dim, 0, batch_size)
    repeats = [1] * len(tensor.shape)
    repeats[dim] = math.ceil(batch_size / tensor.shape[dim])
    return tensor.repeat(*repeats).narrow(dim, 0, batch_size)


def resize_to_batch_size(tensor, batch_size):
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor
    if batch_size <= 1:
        return tensor[:1]
    output = torch.empty(
        [batch_size] + list(tensor.shape[1:]),
        dtype=tensor.dtype, device=tensor.device,
    )
    scale = (in_batch_size - 1) / (batch_size - 1)
    for i in range(batch_size):
        output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    return output


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    out = {}
    for k, v in state_dict.items():
        replaced = False
        for prefix_from, prefix_to in replace_prefix.items():
            if k.startswith(prefix_from):
                out[prefix_to + k[len(prefix_from):]] = v
                replaced = True
                break
        if not replaced and not filter_keys:
            out[k] = v
    return out


def state_dict_key_replace(state_dict, keys_to_replace):
    out = {}
    for k, v in state_dict.items():
        if k in keys_to_replace:
            out[keys_to_replace[k]] = v
        else:
            out[k] = v
    return out


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def weight_dtype(sd, prefix=""):
    dtypes = {}
    for k in sd.keys():
        if k.startswith(prefix):
            d = sd[k].dtype
            dtypes[d] = dtypes.get(d, 0) + sd[k].nelement()
    if not dtypes:
        return torch.float32
    return max(dtypes, key=dtypes.get)


def convert_sd_to(state_dict, dtype):
    return {k: v.to(dtype) if v.is_floating_point() else v for k, v in state_dict.items()}


def safetensors_header(safetensors_path, max_size=100 * 1024 * 1024):
    with open(safetensors_path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        if header_size > max_size:
            return None
        return json.loads(f.read(header_size))


# === Attribute access ===


def get_attr(obj, attr):
    attrs = attr.split(".")
    for a in attrs:
        obj = getattr(obj, a)
    return obj


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for a in attrs[:-1]:
        obj = getattr(obj, a)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], value)
    return prev


def set_attr_param(obj, attr, value):
    attrs = attr.split(".")
    for a in attrs[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))


def copy_to_param(obj, attr, value):
    attrs = attr.split(".")
    for a in attrs[:-1]:
        obj = getattr(obj, a)
    param = getattr(obj, attrs[-1])
    param.data.copy_(value)


# === Progress ===


class ProgressBar:
    def __init__(self, total):
        self.total = total
        self.current = 0

    def update(self, value=1):
        self.current = min(self.current + value, self.total)

    def update_absolute(self, value, total=None):
        if total is not None:
            self.total = total
        self.current = value


# === Misc ===


def bislerp(samples, width, height):
    return torch.nn.functional.interpolate(
        samples, size=(height, width), mode="bilinear", align_corners=False,
    )


def lanczos(samples, width, height):
    return torch.nn.functional.interpolate(
        samples, size=(height, width), mode="bilinear", align_corners=False,
    )


def string_to_seed(s):
    import hashlib
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8,
                upscale_amount=4, out_channels=3, output_device="cpu", pbar=None):
    output = torch.zeros(
        [samples.shape[0], out_channels,
         round(samples.shape[2] * upscale_amount),
         round(samples.shape[3] * upscale_amount)],
        device=output_device,
    )
    out_h = output.shape[2]
    out_w = output.shape[3]
    in_h = samples.shape[2]
    in_w = samples.shape[3]

    rows = math.ceil((in_h - overlap) / (tile_y - overlap)) if tile_y > overlap else 1
    cols = math.ceil((in_w - overlap) / (tile_x - overlap)) if tile_x > overlap else 1

    for row in range(rows):
        for col in range(cols):
            y = min(row * (tile_y - overlap), in_h - tile_y)
            x = min(col * (tile_x - overlap), in_w - tile_x)
            y = max(0, y)
            x = max(0, x)
            tile = samples[:, :, y:y + tile_y, x:x + tile_x]
            result = function(tile)
            oy = round(y * upscale_amount)
            ox = round(x * upscale_amount)
            rh = result.shape[2]
            rw = result.shape[3]
            output[:, :, oy:oy + rh, ox:ox + rw] = result.to(output_device)
            if pbar is not None:
                pbar.update(1)
    return output


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    cols = math.ceil((width - overlap) / (tile_x - overlap)) if tile_x > overlap else 1
    rows = math.ceil((height - overlap) / (tile_y - overlap)) if tile_y > overlap else 1
    return cols * rows
