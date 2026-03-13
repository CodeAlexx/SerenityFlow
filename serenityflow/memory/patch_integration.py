"""Integrates PatchLedger with Stagehand's block transfer pipeline.

Called during H2D transfer in Stagehand's pre-forward hook.
CPU weights never modified. Only GPU copies get patches applied.
"""
from __future__ import annotations

import torch

from serenityflow.core.patch_ledger import PatchEntry, PatchLedger

__all__ = ["apply_patches_to_block"]


def apply_patches_to_block(
    block_name: str,
    gpu_weight: torch.Tensor,
    cpu_weight: torch.Tensor,
    ledger: PatchLedger,
) -> torch.Tensor:
    """Apply all patches for this block to the GPU weight tensor.

    Called by Stagehand during H2D transfer when ledger says block is dirty.

    Args:
        block_name: The block being transferred (e.g., "double_blocks.5")
        gpu_weight: The destination GPU tensor (may contain stale patched data)
        cpu_weight: The clean CPU source tensor (never modified)
        ledger: The model's current PatchLedger

    Returns:
        Patched GPU tensor.
    """
    if not ledger.block_is_dirty(block_name):
        return gpu_weight

    # Start from clean CPU source
    result = cpu_weight.to(gpu_weight.device, non_blocking=True)

    # Apply each patch in order
    for patch in ledger.patches_for_block(block_name):
        result = _apply_single_patch(result, block_name, patch)

    ledger.mark_block_clean(block_name)
    return result


def _apply_single_patch(
    weight: torch.Tensor,
    block_name: str,
    patch: PatchEntry,
) -> torch.Tensor:
    """Apply one patch entry to a weight tensor."""
    matching_keys = [k for k in patch.affected_keys if k.startswith(block_name)]

    for key in matching_keys:
        patch_data = patch.data.get(key)
        if patch_data is None:
            continue

        strength = patch.strength
        applicator = _PATCH_APPLICATORS.get(patch.patch_type)
        if applicator is not None:
            weight = applicator(weight, patch_data, strength)

    return weight


# ─── Patch type applicators ───


def _apply_lora(weight: torch.Tensor, patch_data: dict, strength: float) -> torch.Tensor:
    """Standard LoRA: weight += strength * (up @ down) * alpha/rank"""
    up = patch_data.get("lora_up", patch_data.get("up"))
    down = patch_data.get("lora_down", patch_data.get("down"))
    alpha = patch_data.get("alpha")

    if up is None or down is None:
        return weight

    up = up.to(weight.device, dtype=weight.dtype)
    down = down.to(weight.device, dtype=weight.dtype)

    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item()
        rank = down.shape[0]
        scale = alpha / rank
    else:
        scale = 1.0

    if len(weight.shape) == 4:  # Conv2d
        if len(down.shape) == 4:
            delta = torch.nn.functional.conv2d(
                down.permute(1, 0, 2, 3), up
            ).permute(1, 0, 2, 3)
        else:
            delta = (up @ down).reshape(weight.shape)
    else:  # Linear
        delta = up @ down

    return weight + delta * (strength * scale)


def _apply_loha(weight: torch.Tensor, patch_data: dict, strength: float) -> torch.Tensor:
    """LoHA: Hadamard product of two low-rank decompositions."""
    w1_a = patch_data.get("hada_w1_a")
    w1_b = patch_data.get("hada_w1_b")
    w2_a = patch_data.get("hada_w2_a")
    w2_b = patch_data.get("hada_w2_b")
    alpha = patch_data.get("alpha")

    if w1_a is None or w1_b is None or w2_a is None or w2_b is None:
        return weight

    w1_a = w1_a.to(weight.device, dtype=weight.dtype)
    w1_b = w1_b.to(weight.device, dtype=weight.dtype)
    w2_a = w2_a.to(weight.device, dtype=weight.dtype)
    w2_b = w2_b.to(weight.device, dtype=weight.dtype)

    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item()
        rank = w1_b.shape[0]
        scale = alpha / rank
    else:
        scale = 1.0

    delta = (w1_a @ w1_b) * (w2_a @ w2_b)

    if len(weight.shape) == 4 and len(delta.shape) == 2:
        delta = delta.reshape(weight.shape)

    return weight + delta * (strength * scale)


def _apply_lokr(weight: torch.Tensor, patch_data: dict, strength: float) -> torch.Tensor:
    """LoKr: Kronecker product decomposition."""
    w1 = patch_data.get("lokr_w1")
    w2 = patch_data.get("lokr_w2")
    w1_a = patch_data.get("lokr_w1_a")
    w1_b = patch_data.get("lokr_w1_b")
    w2_a = patch_data.get("lokr_w2_a")
    w2_b = patch_data.get("lokr_w2_b")
    alpha = patch_data.get("alpha")

    if w1 is None and w1_a is None:
        return weight

    if w1_a is not None and w1_b is not None:
        w1 = w1_a.to(weight.device, dtype=weight.dtype) @ w1_b.to(weight.device, dtype=weight.dtype)
    elif w1 is not None:
        w1 = w1.to(weight.device, dtype=weight.dtype)
    else:
        return weight

    if w2_a is not None and w2_b is not None:
        w2 = w2_a.to(weight.device, dtype=weight.dtype) @ w2_b.to(weight.device, dtype=weight.dtype)
    elif w2 is not None:
        w2 = w2.to(weight.device, dtype=weight.dtype)
    else:
        return weight

    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item()
        rank = w1_b.shape[0] if w1_b is not None else w1.shape[0]
        scale = alpha / rank
    else:
        scale = 1.0

    delta = torch.kron(w1, w2)
    if len(weight.shape) == 4 and len(delta.shape) == 2:
        delta = delta.reshape(weight.shape)

    return weight + delta * (strength * scale)


def _apply_ia3(weight: torch.Tensor, patch_data: dict, strength: float) -> torch.Tensor:
    """IA3: element-wise scaling."""
    ia3_weight = patch_data.get("weight", patch_data.get("ia3_weight"))
    if ia3_weight is None:
        return weight
    ia3_weight = ia3_weight.to(weight.device, dtype=weight.dtype)
    return weight * (1.0 + strength * ia3_weight)


def _apply_full_diff(weight: torch.Tensor, patch_data: dict, strength: float) -> torch.Tensor:
    """Full diff: direct weight delta."""
    diff = patch_data.get("diff")
    if diff is None:
        return weight
    diff = diff.to(weight.device, dtype=weight.dtype)
    return weight + diff * strength


def _apply_glora(weight: torch.Tensor, patch_data: dict, strength: float) -> torch.Tensor:
    """GLoRA: generalized LoRA with additional scaling matrices."""
    a1 = patch_data.get("a1")
    a2 = patch_data.get("a2")
    b1 = patch_data.get("b1")
    b2 = patch_data.get("b2")

    if a1 is None or a2 is None or b1 is None or b2 is None:
        return weight

    a1 = a1.to(weight.device, dtype=weight.dtype)
    a2 = a2.to(weight.device, dtype=weight.dtype)
    b1 = b1.to(weight.device, dtype=weight.dtype)
    b2 = b2.to(weight.device, dtype=weight.dtype)

    delta = (a1 @ b1) + (a2 @ b2)
    return weight + delta * strength


def _apply_ortho(weight: torch.Tensor, patch_data: dict, strength: float) -> torch.Tensor:
    """OrthoLoRA: same compute as standard LoRA at inference."""
    return _apply_lora(weight, patch_data, strength)


_PATCH_APPLICATORS = {
    "lora": _apply_lora,
    "loha": _apply_loha,
    "lokr": _apply_lokr,
    "ia3": _apply_ia3,
    "full_diff": _apply_full_diff,
    "glora": _apply_glora,
    "ortho": _apply_ortho,
}
