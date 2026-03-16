"""Resolve model filenames to filesystem paths.

Compatible with ComfyUI's model directory structure.
Supports extra_model_paths.yaml.
Integrates with Stagehand's unified model resolver (~/.serenity/models/).
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified model resolver integration (optional dependency)
# ---------------------------------------------------------------------------

try:
    from stagehand.model_resolver import get_resolver as _get_unified_resolver
    _HAS_UNIFIED = True
except ImportError:
    _HAS_UNIFIED = False

# Map SerenityFlow folder names → unified ModelType subdirectory names
_UNIFIED_TYPE_MAP: dict[str, str | None] = {
    "checkpoints": "checkpoints",
    "diffusion_models": "checkpoints",  # Same thing in unified layout
    "loras": "loras",
    "vae": "vaes",
    "clip": "text_encoders",
    "controlnet": "controlnets",
    "clip_vision": "text_encoders",
    "upscale_models": "upscalers",
    "embeddings": None,  # No unified equivalent yet
    "style_models": None,
}

_UNIFIED_BASE = os.path.expanduser("~/.serenity/models")

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

# Directory name aliases (matches folder_paths compat shim)
_DIR_ALIASES: dict[str, str] = {
    "stable-diffusion": "checkpoints",
    "lora": "loras",
    "embedding": "embeddings",
    "Lora": "loras",
    "Embeddings": "embeddings",
    "ltx2": "diffusion_models",
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
            paths: list[str] = []

            # Prepend unified ~/.serenity/models/<type> dir as FIRST search path
            unified_type = _UNIFIED_TYPE_MAP.get(folder)
            if unified_type is not None:
                unified_dir = os.path.join(_UNIFIED_BASE, unified_type)
                paths.append(unified_dir)

            for sd in subdirs:
                paths.append(os.path.join(base_dir, sd))
                # Also try the leaf directory directly under base_dir
                leaf = os.path.basename(sd)
                direct = os.path.join(base_dir, leaf)
                if direct not in paths:
                    paths.append(direct)
            # Collect alias names that map to this category
            alias_names = {a for a, cat in _DIR_ALIASES.items() if cat == folder}

            # Case-insensitive: probe actual directory entries for matches
            if os.path.isdir(base_dir):
                folder_names = {os.path.basename(sd) for sd in subdirs}
                folder_names.add(folder)  # also match the key itself
                folder_names |= alias_names  # include aliases (e.g. Stable-Diffusion)
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

        Checks the unified Stagehand resolver first (fast index lookup),
        then falls back to the legacy directory walk.
        Returns full path. Raises FileNotFoundError if not found.
        """
        # 1. Try unified resolver first (fast index lookup)
        if _HAS_UNIFIED:
            unified_type = _UNIFIED_TYPE_MAP.get(folder)
            if unified_type is not None:
                try:
                    resolver = _get_unified_resolver()
                    path = resolver.resolve_file(filename, unified_type)
                    return str(path)
                except FileNotFoundError:
                    pass
                # Also try just the stem (name without extension)
                stem = Path(filename).stem
                if stem != filename:
                    try:
                        path = resolver.resolve_file(stem, unified_type)
                        return str(path)
                    except FileNotFoundError:
                        pass

        # 2. Fall back to existing directory walk
        search_dirs = self.dirs.get(folder, [])
        for search_dir in search_dirs:
            full_path = os.path.join(search_dir, filename)
            if os.path.exists(full_path):
                return full_path
            # Also check subdirectories one level deep
            if os.path.isdir(search_dir):
                try:
                    for entry in os.scandir(search_dir):
                        if entry.is_dir():
                            sub_path = os.path.join(entry.path, filename)
                            if os.path.exists(sub_path):
                                return sub_path
                except OSError:
                    pass

        raise FileNotFoundError(
            f"Model '{filename}' not found in {folder} paths: {search_dirs}"
        )

    def list_models(self, folder: str) -> list[str]:
        """List all model files in a folder's search paths.

        Includes models from the unified Stagehand index when available.
        """
        models: list[str] = []
        seen: set[str] = set()

        # Include models from unified resolver index
        if _HAS_UNIFIED:
            unified_type = _UNIFIED_TYPE_MAP.get(folder)
            if unified_type is not None:
                try:
                    resolver = _get_unified_resolver()
                    for entry in resolver.list_models(unified_type):
                        for fname in entry.files:
                            if fname not in seen:
                                seen.add(fname)
                                models.append(fname)
                except Exception:
                    pass

        # Legacy directory walk
        search_dirs = self.dirs.get(folder, [])
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


_WELL_KNOWN_MODEL_DIRS = [
    _UNIFIED_BASE,
    os.path.expanduser("~/EriDiffusion/Models"),
    os.path.expanduser("~/eriui/comfyui/ComfyUI/models"),
    os.path.expanduser("~/SwarmUI/Models"),
]


def _detect_base_dir() -> str:
    """Auto-detect model base directory, matching folder_paths logic."""
    for d in _WELL_KNOWN_MODEL_DIRS:
        if os.path.isdir(d):
            return d
    return os.getcwd()


def _count_unified_models() -> int:
    """Count models available through the unified resolver. Returns 0 on error."""
    if not _HAS_UNIFIED:
        return 0
    try:
        return len(_get_unified_resolver().list_models())
    except Exception:
        return 0


def get_model_paths(base_dir: str | None = None) -> ModelPaths:
    """Get or create the global ModelPaths instance.

    On first call, base_dir is auto-detected if not provided.
    Also merges paths from all well-known model directories.
    Subsequent calls return the cached instance (ignoring base_dir).
    """
    global _instance
    if _instance is None:
        if base_dir is None:
            base_dir = _detect_base_dir()
        _instance = ModelPaths(base_dir)
        # Merge paths from other well-known model directories
        for extra_dir in _WELL_KNOWN_MODEL_DIRS:
            if os.path.isdir(extra_dir) and extra_dir != base_dir:
                extra = ModelPaths(extra_dir)
                for folder, paths in extra.dirs.items():
                    existing = _instance.dirs.setdefault(folder, [])
                    for p in paths:
                        if p not in existing:
                            existing.append(p)
    return _instance


def set_model_paths(paths: ModelPaths) -> None:
    """Replace the global ModelPaths instance."""
    global _instance
    _instance = paths


__all__ = [
    "ModelPaths",
    "DEFAULT_MODEL_DIRS",
    "get_model_paths",
    "set_model_paths",
    "_UNIFIED_TYPE_MAP",
    "_HAS_UNIFIED",
    "_count_unified_models",
]
