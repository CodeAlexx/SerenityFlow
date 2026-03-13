from __future__ import annotations

"""
Tracks all model modifications — LoRA, adapters, weight patches.
CPU weights never modified. Only GPU copies during Stagehand H2D transfer.
Epoch-based dirty tracking per block.
"""
import hashlib
from dataclasses import dataclass, field

@dataclass
class PatchEntry:
    patch_type: str         # lora, loha, lokr, glora, ia3, full_diff, ortho
    source: str             # File path or identifier
    strength: float
    affected_keys: set      # Weight keys this patch touches
    data: dict              # Actual patch tensors (or references)

class PatchLedger:
    def __init__(self):
        self.epoch: int = 0
        self.patches: list[PatchEntry] = []
        self.block_stamps: dict[str, int] = {}  # block_name -> epoch when last applied
        self._fingerprint_cache: str | None = None

    def add_patch(self, entry: PatchEntry) -> None:
        self.patches.append(entry)
        self.epoch += 1
        self._fingerprint_cache = None

    def remove_patch(self, source: str) -> None:
        before = len(self.patches)
        self.patches = [p for p in self.patches if p.source != source]
        if len(self.patches) != before:
            self.epoch += 1
            self._fingerprint_cache = None

    def clear(self) -> None:
        if self.patches:
            self.patches.clear()
            self.epoch += 1
            self._fingerprint_cache = None

    def fingerprint(self) -> str:
        if self._fingerprint_cache is not None:
            return self._fingerprint_cache
        h = hashlib.sha256()
        for p in self.patches:
            h.update(p.patch_type.encode())
            h.update(p.source.encode())
            h.update(str(p.strength).encode())
        self._fingerprint_cache = h.hexdigest()[:16]
        return self._fingerprint_cache

    def patches_for_block(self, block_name: str) -> list[PatchEntry]:
        result = []
        for p in self.patches:
            for key in p.affected_keys:
                if key.startswith(block_name) or block_name.startswith(key.rsplit('.', 1)[0] if '.' in key else key):
                    result.append(p)
                    break
        return result

    def block_is_dirty(self, block_name: str) -> bool:
        return self.block_stamps.get(block_name, -1) < self.epoch

    def mark_block_clean(self, block_name: str) -> None:
        self.block_stamps[block_name] = self.epoch

    def clone(self) -> PatchLedger:
        new = PatchLedger()
        new.patches = list(self.patches)  # Shared PatchEntry refs
        new.epoch = self.epoch
        # block_stamps intentionally NOT copied — new instance, all blocks dirty
        return new
