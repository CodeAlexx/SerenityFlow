"""PatchLedger: Coordinates LoRA patching with Stagehand block residency.

Cleaned from v1 -- NO comfy imports. Accepts any model with a patches dict.

One PatchLedger per model. Keeps CPU backups of clean (un-patched) weights
so LoRA can be applied from a known-clean baseline after block eviction/reload.
"""
from __future__ import annotations

import logging
import weakref

import torch

logger = logging.getLogger(__name__)

__all__ = ["PatchLedger"]

_BACKUP_ATTR = "_stagehand_clean_backups"


def _get_or_create_backups(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get or create the clean-weight backup dict on the model."""
    if not hasattr(model, _BACKUP_ATTR):
        setattr(model, _BACKUP_ATTR, {})
    return getattr(model, _BACKUP_ATTR)


class PatchLedger:
    """Coordinates LoRA patching with Stagehand block residency.

    Unlike v1, this does not depend on ComfyUI's ModelPatcher or lora module.
    Instead it accepts:
    - model: the nn.Module being patched
    - patches: dict of {param_key: patch_data} (managed externally)
    - apply_fn: callable(param_path, clean_weight, patch_data) -> patched_weight
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patches: dict | None = None,
        apply_fn=None,
        block_prefix: str = "",
    ) -> None:
        self._model_ref: weakref.ref[torch.nn.Module] = weakref.ref(model)
        self._patches = patches or {}
        self._apply_fn = apply_fn
        self._block_prefix = block_prefix
        self.patch_epoch: int = 0
        self.block_epochs: dict[str, int] = {}
        self._block_patch_keys: dict[str, list[str]] = {}
        self._keys_cached_at_epoch: int = -1

    def set_patches(self, patches: dict):
        """Update the patches dict and bump epoch."""
        self._patches = patches
        self.increment_epoch()

    def increment_epoch(self) -> None:
        """Called when patches change."""
        self.patch_epoch += 1

    def on_block_loaded(
        self,
        block_id: str,
        module: torch.nn.Module,
        was_resident: bool,
    ) -> None:
        """Called in pre-forward hook after Stagehand ensures block is on GPU."""
        if not was_resident:
            self.block_epochs.pop(block_id, None)

        if self.block_epochs.get(block_id) == self.patch_epoch:
            return

        self._apply_patches_to_block(block_id, module)
        self.block_epochs[block_id] = self.patch_epoch

    def has_patches(self) -> bool:
        return len(self._patches) > 0

    @staticmethod
    def restore_clean_weights(model: torch.nn.Module) -> None:
        """Restore all backed-up weights to their clean state."""
        backups = getattr(model, _BACKUP_ATTR, None)
        if not backups:
            return

        restored = 0
        for param_path, clean_cpu in backups.items():
            parts = param_path.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is None:
                continue
            param = getattr(obj, parts[-1], None)
            if param is None or not isinstance(param, (torch.nn.Parameter, torch.Tensor)):
                continue
            weight = param if isinstance(param, torch.Tensor) else param.data
            clean_on_device = clean_cpu.to(weight.device, non_blocking=True)
            weight.data.copy_(clean_on_device)
            restored += 1

        logger.info(
            "PatchLedger: Restored %d/%d clean weights on %s",
            restored, len(backups), model.__class__.__name__,
        )
        backups.clear()

    def _get_patch_keys_for_block(self, block_id: str) -> list[str]:
        """Return patch keys belonging to block_id, with caching."""
        if self._keys_cached_at_epoch != self.patch_epoch:
            self._block_patch_keys.clear()
            self._keys_cached_at_epoch = self.patch_epoch

        if block_id in self._block_patch_keys:
            return self._block_patch_keys[block_id]

        prefix = f"{self._block_prefix}{block_id}."
        keys = [k for k in self._patches if k.startswith(prefix)]
        self._block_patch_keys[block_id] = keys
        return keys

    def _apply_patches_to_block(
        self, block_id: str, module: torch.nn.Module
    ) -> None:
        """Apply patches to weights in this block using apply_fn."""
        model = self._model_ref()
        if model is None or self._apply_fn is None:
            return

        keys = self._get_patch_keys_for_block(block_id)
        if not keys:
            return

        backups = _get_or_create_backups(model)
        applied = 0

        for key in keys:
            if key not in self._patches:
                continue

            # Navigate to the parameter
            param_path = key[len(self._block_prefix):] if key.startswith(self._block_prefix) else key
            parts = param_path.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is None:
                continue

            param = getattr(obj, parts[-1], None)
            if param is None or not isinstance(param, (torch.nn.Parameter, torch.Tensor)):
                continue

            weight = param if isinstance(param, torch.Tensor) else param.data

            # Save clean backup on first encounter
            if param_path not in backups:
                backups[param_path] = weight.data.to("cpu", copy=True)

            # Apply patch from clean baseline
            clean_weight = backups[param_path].to(weight.device)
            patched = self._apply_fn(param_path, clean_weight, self._patches[key])
            weight.data.copy_(patched)
            applied += 1

        if applied > 0:
            logger.debug(
                "PatchLedger: Applied %d/%d patches to block %s (epoch=%d)",
                applied, len(keys), block_id, self.patch_epoch,
            )
