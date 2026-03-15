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

# Auto-detect model directory: prefer ~/EriDiffusion/Models if it exists
_eri_models = os.path.expanduser("~/EriDiffusion/Models")
_auto_detect_models = not os.path.isdir(models_dir) and os.path.isdir(_eri_models)

# Category → directory paths
folder_names_and_paths: dict[str, tuple[list[str], set[str]]] = {}

# File extensions per category
_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
_MODEL_EXTENSIONS_GGUF = _MODEL_EXTENSIONS | {".gguf"}

# Default categories
_DEFAULT_CATEGORIES = {
    "checkpoints": ([os.path.join(models_dir, "checkpoints")], _MODEL_EXTENSIONS),
    "configs": ([os.path.join(models_dir, "configs")], {".yaml", ".json"}),
    "loras": ([os.path.join(models_dir, "loras")], {".safetensors", ".ckpt", ".pt", ".pth"}),
    "vae": ([os.path.join(models_dir, "vae")], _MODEL_EXTENSIONS),
    "clip": ([os.path.join(models_dir, "clip")], _MODEL_EXTENSIONS_GGUF),
    "clip_vision": ([os.path.join(models_dir, "clip_vision")], _MODEL_EXTENSIONS),
    "style_models": ([os.path.join(models_dir, "style_models")], _MODEL_EXTENSIONS),
    "embeddings": ([os.path.join(models_dir, "embeddings")], _MODEL_EXTENSIONS),
    "diffusers": ([os.path.join(models_dir, "diffusers")], set()),
    "diffusion_models": ([os.path.join(models_dir, "diffusion_models")], _MODEL_EXTENSIONS_GGUF),
    "controlnet": ([os.path.join(models_dir, "controlnet")], _MODEL_EXTENSIONS),
    "gligen": ([os.path.join(models_dir, "gligen")], _MODEL_EXTENSIONS),
    "upscale_models": ([os.path.join(models_dir, "upscale_models")], _MODEL_EXTENSIONS),
    "hypernetworks": ([os.path.join(models_dir, "hypernetworks")], _MODEL_EXTENSIONS),
    "unet": ([os.path.join(models_dir, "unet")], _MODEL_EXTENSIONS_GGUF),
    "text_encoders": ([os.path.join(models_dir, "text_encoders")], _MODEL_EXTENSIONS_GGUF),
}

# Aliases: directory names that map to category names (case-insensitive matching)
_DIR_ALIASES = {
    "stable-diffusion": "checkpoints",
    "lora": "loras",
    "embedding": "embeddings",
    "Lora": "loras",
    "Embeddings": "embeddings",
    "ltx2": "diffusion_models",
}

# Cross-register: checkpoints and diffusion_models share search paths
# Many setups (ComfyUI, EriDiffusion) put full models in diffusion_models/
_CROSS_CATEGORIES = [
    ("checkpoints", "diffusion_models"),
]

for cat, (paths, exts) in _DEFAULT_CATEGORIES.items():
    folder_names_and_paths[cat] = (list(paths), set(exts))

# Cross-register shared search paths between related categories
for cat_a, cat_b in _CROSS_CATEGORIES:
    if cat_a in folder_names_and_paths and cat_b in folder_names_and_paths:
        paths_a = folder_names_and_paths[cat_a][0]
        paths_b = folder_names_and_paths[cat_b][0]
        for p in paths_b:
            if p not in paths_a:
                paths_a.append(p)
        for p in paths_a:
            if p not in paths_b:
                paths_b.append(p)


def set_base_path(new_base: str):
    """Reinitialize all model paths from a new base directory.

    Called when --model-dir is specified at startup.
    Supports two layouts:
      - Nested: base/models/checkpoints/  (ComfyUI default)
      - Flat:   base/checkpoints/         (symlink layouts)
    Also handles directory name aliases (Stable-Diffusion → checkpoints, etc.)
    """
    global base_path, models_dir, output_directory, temp_directory, input_directory

    base_path = new_base
    models_dir = os.path.join(base_path, "models")
    output_directory = os.path.join(base_path, "output")
    temp_directory = os.path.join(base_path, "temp")
    input_directory = os.path.join(base_path, "input")

    # Rebuild folder_names_and_paths with new models_dir (nested layout)
    for cat, (_, exts) in _DEFAULT_CATEGORIES.items():
        new_paths = [os.path.join(models_dir, cat)]
        folder_names_and_paths[cat] = (new_paths, set(exts))

    # Scan base_path for flat layout dirs and aliases
    if os.path.isdir(base_path):
        # Build reverse lookup: lowercase dir name → category
        cat_lookup = {}
        for cat in _DEFAULT_CATEGORIES:
            cat_lookup[cat.lower()] = cat
        for alias, cat in _DIR_ALIASES.items():
            cat_lookup[alias.lower()] = cat

        try:
            for entry in os.scandir(base_path):
                if entry.is_dir() or entry.is_symlink():
                    resolved = entry.name.lower()
                    cat = cat_lookup.get(resolved)
                    if cat and cat in folder_names_and_paths:
                        real_path = os.path.realpath(entry.path)
                        paths = folder_names_and_paths[cat][0]
                        if entry.path not in paths and real_path not in paths:
                            paths.append(entry.path)
        except OSError:
            pass


# Apply auto-detected model path (after set_base_path is defined)
if _auto_detect_models:
    set_base_path(_eri_models)

# Also scan well-known model directories and add them
_WELL_KNOWN_MODEL_DIRS = [
    os.path.expanduser("~/EriDiffusion/Models"),
    os.path.expanduser("~/eriui/comfyui/ComfyUI/models"),
    os.path.expanduser("~/SwarmUI/Models"),
]

def add_extra_model_dirs(extra_dirs: list[str] | None = None):
    """Add additional model search directories.

    Scans each directory for category subdirectories (checkpoints, loras, etc.)
    and adds them to the search paths.
    """
    dirs_to_scan = list(extra_dirs or []) + _WELL_KNOWN_MODEL_DIRS
    cat_lookup = {}
    for cat in _DEFAULT_CATEGORIES:
        cat_lookup[cat.lower()] = cat
    for alias, cat in _DIR_ALIASES.items():
        cat_lookup[alias.lower()] = cat

    for base in dirs_to_scan:
        base = os.path.expanduser(base)
        if not os.path.isdir(base):
            continue
        try:
            for entry in os.scandir(base):
                if not (entry.is_dir() or entry.is_symlink()):
                    continue
                cat = cat_lookup.get(entry.name.lower())
                if cat and cat in folder_names_and_paths:
                    real_path = os.path.realpath(entry.path)
                    paths = folder_names_and_paths[cat][0]
                    if entry.path not in paths and real_path not in paths:
                        paths.append(entry.path)
        except OSError:
            pass

# Auto-add well-known dirs on import
add_extra_model_dirs()


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
                    full = os.path.join(root, f)
                    # Skip broken symlinks
                    if os.path.islink(full) and not os.path.exists(full):
                        continue
                    rel = os.path.relpath(full, path)
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
