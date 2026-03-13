"""Resolve model filenames to filesystem paths.

Compatible with ComfyUI's model directory structure.
Supports extra_model_paths.yaml.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIRS = {
    "checkpoints": ["models/checkpoints"],
    "diffusion_models": ["models/unet", "models/diffusion_models"],
    "loras": ["models/loras"],
    "vae": ["models/vae"],
    "clip": ["models/clip", "models/text_encoders"],
    "controlnet": ["models/controlnet"],
    "clip_vision": ["models/clip_vision"],
    "style_models": ["models/style_models"],
    "upscale_models": ["models/upscale_models"],
    "embeddings": ["models/embeddings"],
}


class ModelPaths:
    def __init__(self, base_dir: str, extra_paths_yaml: str | None = None):
        """
        base_dir: root directory (where models/ lives)
        extra_paths_yaml: optional path to extra_model_paths.yaml
        """
        self.base_dir = base_dir
        self.dirs: dict[str, list[str]] = {}

        # Load defaults -- try both "base_dir/models/X" and "base_dir/X" layouts
        # Also handle case-insensitive directory names (VAE vs vae, Lora vs loras)
        for folder, subdirs in DEFAULT_MODEL_DIRS.items():
            paths = []
            for sd in subdirs:
                paths.append(os.path.join(base_dir, sd))
                # Also try the leaf directory directly under base_dir
                leaf = os.path.basename(sd)
                direct = os.path.join(base_dir, leaf)
                if direct not in paths:
                    paths.append(direct)
            # Case-insensitive: probe actual directory entries for matches
            if os.path.isdir(base_dir):
                folder_names = {os.path.basename(sd) for sd in subdirs}
                folder_names.add(folder)  # also match the key itself
                try:
                    for entry in os.scandir(base_dir):
                        if entry.is_dir() and entry.name.lower() in {
                            n.lower() for n in folder_names
                        }:
                            if entry.path not in paths:
                                paths.append(entry.path)
                except OSError:
                    pass
            self.dirs[folder] = paths

        # Load extra paths
        if extra_paths_yaml and os.path.exists(extra_paths_yaml):
            self._load_extra_paths(extra_paths_yaml)

    def find(self, filename: str, folder: str = "checkpoints") -> str:
        """Find model file by name in the given folder's search paths.
        Returns full path. Raises FileNotFoundError if not found.
        """
        search_dirs = self.dirs.get(folder, [])
        for search_dir in search_dirs:
            full_path = os.path.join(search_dir, filename)
            if os.path.exists(full_path):
                return full_path
            # Also check subdirectories one level deep
            if os.path.isdir(search_dir):
                for entry in os.scandir(search_dir):
                    if entry.is_dir():
                        sub_path = os.path.join(entry.path, filename)
                        if os.path.exists(sub_path):
                            return sub_path

        raise FileNotFoundError(
            f"Model '{filename}' not found in {folder} paths: {search_dirs}"
        )

    def list_models(self, folder: str) -> list[str]:
        """List all model files in a folder's search paths."""
        search_dirs = self.dirs.get(folder, [])
        models = []
        seen = set()
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            for root, dirs, files in os.walk(search_dir):
                for fname in files:
                    if fname.startswith("."):
                        continue
                    # Return path relative to search_dir
                    rel = os.path.relpath(os.path.join(root, fname), search_dir)
                    if rel not in seen:
                        seen.add(rel)
                        models.append(rel)
        return sorted(models)

    def _load_extra_paths(self, yaml_path: str):
        """Parse ComfyUI's extra_model_paths.yaml format."""
        try:
            import yaml
        except ImportError:
            log.warning("PyYAML not installed, cannot load extra_model_paths.yaml")
            return

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return

        for section_name, section in data.items():
            if not isinstance(section, dict):
                continue
            base_path = section.get("base_path", "")
            for folder, path in section.items():
                if folder == "base_path":
                    continue
                if isinstance(path, str):
                    full_path = os.path.join(base_path, path) if base_path else path
                    self.dirs.setdefault(folder, []).append(full_path)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: ModelPaths | None = None


def get_model_paths(base_dir: str | None = None) -> ModelPaths:
    """Get or create the global ModelPaths instance.

    On first call, base_dir defaults to the current working directory.
    Subsequent calls return the cached instance (ignoring base_dir).
    """
    global _instance
    if _instance is None:
        if base_dir is None:
            base_dir = os.getcwd()
        _instance = ModelPaths(base_dir)
    return _instance


def set_model_paths(paths: ModelPaths) -> None:
    """Replace the global ModelPaths instance."""
    global _instance
    _instance = paths


__all__ = ["ModelPaths", "DEFAULT_MODEL_DIRS", "get_model_paths", "set_model_paths"]
