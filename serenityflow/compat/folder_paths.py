"""Compatibility shim for folder_paths.

Model directory resolution with ComfyUI-compatible category names.
"""
from __future__ import annotations

import os
from typing import Optional

# Base directories — set during startup
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_path, "models")
output_directory = os.path.join(base_path, "output")
temp_directory = os.path.join(base_path, "temp")
input_directory = os.path.join(base_path, "input")

# Category → directory paths
folder_names_and_paths: dict[str, tuple[list[str], set[str]]] = {}

# Default categories
_DEFAULT_CATEGORIES = {
    "checkpoints": ([os.path.join(models_dir, "checkpoints")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "configs": ([os.path.join(models_dir, "configs")], {".yaml", ".json"}),
    "loras": ([os.path.join(models_dir, "loras")], {".safetensors", ".ckpt", ".pt", ".pth"}),
    "vae": ([os.path.join(models_dir, "vae")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "clip": ([os.path.join(models_dir, "clip")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "clip_vision": ([os.path.join(models_dir, "clip_vision")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "style_models": ([os.path.join(models_dir, "style_models")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "embeddings": ([os.path.join(models_dir, "embeddings")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "diffusers": ([os.path.join(models_dir, "diffusers")], set()),
    "controlnet": ([os.path.join(models_dir, "controlnet")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "gligen": ([os.path.join(models_dir, "gligen")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "upscale_models": ([os.path.join(models_dir, "upscale_models")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "hypernetworks": ([os.path.join(models_dir, "hypernetworks")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}),
    "unet": ([os.path.join(models_dir, "unet")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}),
    "text_encoders": ([os.path.join(models_dir, "text_encoders")], {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}),
}

for cat, (paths, exts) in _DEFAULT_CATEGORIES.items():
    folder_names_and_paths[cat] = (list(paths), set(exts))


def set_output_directory(directory: str):
    global output_directory
    output_directory = directory


def set_temp_directory(directory: str):
    global temp_directory
    temp_directory = directory


def set_input_directory(directory: str):
    global input_directory
    input_directory = directory


def get_output_directory():
    os.makedirs(output_directory, exist_ok=True)
    return output_directory


def get_temp_directory():
    os.makedirs(temp_directory, exist_ok=True)
    return temp_directory


def get_input_directory():
    os.makedirs(input_directory, exist_ok=True)
    return input_directory


def get_folder_paths(folder_name: str) -> list[str]:
    if folder_name in folder_names_and_paths:
        return folder_names_and_paths[folder_name][0]
    return []


def get_full_path(folder_name: str, filename: str) -> Optional[str]:
    paths = get_folder_paths(folder_name)
    for path in paths:
        full = os.path.join(path, filename)
        if os.path.exists(full):
            return full
    # Return first path + filename even if doesn't exist
    if paths:
        return os.path.join(paths[0], filename)
    return None


def get_full_path_or_raise(folder_name: str, filename: str) -> str:
    result = get_full_path(folder_name, filename)
    if result is None or not os.path.exists(result):
        raise FileNotFoundError(f"Could not find {filename} in {folder_name} paths: {get_folder_paths(folder_name)}")
    return result


def get_filename_list(folder_name: str) -> list[str]:
    output = []
    paths = get_folder_paths(folder_name)
    extensions = set()
    if folder_name in folder_names_and_paths:
        extensions = folder_names_and_paths[folder_name][1]

    for path in paths:
        if not os.path.isdir(path):
            continue
        for root, dirs, files in os.walk(path):
            for f in files:
                if not extensions or os.path.splitext(f)[1].lower() in extensions:
                    rel = os.path.relpath(os.path.join(root, f), path)
                    output.append(rel)
    return sorted(set(output))


def get_filename_list_(folder_name: str) -> tuple[list[str], dict]:
    return get_filename_list(folder_name), {}


def add_model_folder_path(folder_name: str, full_folder_path: str, is_default: bool = False):
    if folder_name not in folder_names_and_paths:
        folder_names_and_paths[folder_name] = ([], set())
    paths = folder_names_and_paths[folder_name][0]
    if full_folder_path not in paths:
        if is_default:
            paths.insert(0, full_folder_path)
        else:
            paths.append(full_folder_path)


def get_save_image_path(filename_prefix: str, output_dir: str, image_width=0,
                         image_height=0) -> tuple[str, str, int, str, str]:
    full_output_folder = os.path.join(output_dir, os.path.dirname(filename_prefix))
    os.makedirs(full_output_folder, exist_ok=True)

    filename = os.path.basename(filename_prefix)

    # Find next counter
    counter = 1
    if os.path.isdir(full_output_folder):
        existing = [f for f in os.listdir(full_output_folder)
                    if f.startswith(filename) and "_" in f]
        for f in existing:
            try:
                num = int(f.split("_")[-1].split(".")[0])
                counter = max(counter, num + 1)
            except (ValueError, IndexError):
                pass

    return full_output_folder, filename, counter, "", ""


def get_annotated_filepath(name: str, default_dir: Optional[str] = None) -> str:
    if os.path.isabs(name):
        return name
    if default_dir is not None:
        return os.path.join(default_dir, name)
    return os.path.join(input_directory, name)


def exists_annotated_filepath(name: str) -> bool:
    return os.path.exists(get_annotated_filepath(name))


def map_filename(filename: str) -> tuple[str, Optional[str]]:
    return filename, None


supported_pt_extensions = {".ckpt", ".pt", ".bin", ".pth", ".safetensors"}
