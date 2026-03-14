"""Stagehand block-swap coordinator for SerenityFlow v2.

Cleaned from v1 -- NO comfy imports. Config passed explicitly.
Owns shared PinnedPool, creates StagehandRuntime instances for qualifying models.
"""
from __future__ import annotations

import logging
import re
import threading
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Optional

import torch

try:
    from stagehand import BlockState, PinnedPool, StagehandConfig, StagehandRuntime
except ImportError:
    BlockState = None
    PinnedPool = None
    StagehandConfig = None
    StagehandRuntime = None

if TYPE_CHECKING:
    from serenityflow.memory.graph_planner import GraphSchedule

logger = logging.getLogger(__name__)

# Multi-architecture block patterns
BLOCK_PATTERNS: dict[str, dict] = {
    "flux": {
        "patterns": [r"^double_blocks\.\d+$", r"^single_blocks\.\d+$"],
        "min_blocks": 10,
        "require_all": True,
    },
    "hunyuan_video": {
        "patterns": [r"^double_blocks\.\d+$", r"^single_blocks\.\d+$"],
        "min_blocks": 10,
        "require_all": True,
    },
    "sdxl": {
        "patterns": [r"^input_blocks\.\d+$", r"^output_blocks\.\d+$"],
        "min_blocks": 6,
        "require_all": True,
    },
    "sd3": {
        "patterns": [r"^joint_blocks\.\d+$"],
        "min_blocks": 6,
        "require_all": False,
    },
    "wan": {
        "patterns": [r"^blocks\.\d+$"],
        "min_blocks": 10,
        "require_all": False,
    },
    "ltxv": {
        "patterns": [r"^transformer_blocks\.\d+$"],
        "min_blocks": 6,
        "require_all": False,
    },
}


def _detect_architecture(model: torch.nn.Module) -> tuple[str | None, str | None, int]:
    """Detect model architecture by matching block patterns."""
    module_names: list[str] = [name for name, _ in model.named_modules() if name]

    for arch_name, spec in BLOCK_PATTERNS.items():
        patterns = [re.compile(p) for p in spec["patterns"]]
        require_all = spec.get("require_all", False)
        min_blocks = spec["min_blocks"]

        counts: dict[str, int] = {}
        for pat in patterns:
            key = pat.pattern
            for name in module_names:
                if pat.search(name):
                    counts[key] = counts.get(key, 0) + 1

        if require_all and len(counts) < len(patterns):
            continue

        total = sum(counts.values())
        if total < min_blocks:
            continue

        inner_parts = []
        for p in spec["patterns"]:
            stripped = p.lstrip("^").rstrip("$")
            inner_parts.append(stripped)
        runtime_pattern = "^(" + "|".join(inner_parts) + ")$"

        logger.info(
            "Stagehand: Detected %s architecture (%d blocks: %s)",
            arch_name, total, counts,
        )
        return arch_name, runtime_pattern, total

    return None, None, 0


def _model_size_mb(model: torch.nn.Module) -> int:
    """Estimate total model parameter size in MB."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes // (1024 * 1024)


class StagehandCoordinator:
    """Owns shared PinnedPool and creates StagehandRuntime instances."""

    def __init__(
        self,
        pool_mb: int | None = None,
        vram_budget_mb: int | None = None,
        prefetch_window: int = 3,
        telemetry: bool = False,
        block_threshold_mb: int = 2048,
    ):
        if PinnedPool is None:
            raise ImportError("Stagehand not installed. Install with: pip install stagehand")

        # Auto-size pinned pool
        if pool_mb is None:
            try:
                import psutil
                total_ram = psutil.virtual_memory().total
                pool_mb = min(8192, total_ram // (4 * 1024 * 1024))
            except ImportError:
                pool_mb = 4096
                logger.warning("psutil not available, using default pool size %d MB", pool_mb)

        # Auto-size VRAM budget
        if vram_budget_mb is None:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                vram_budget_mb = int(props.total_memory * 0.9 / (1024 * 1024))
            else:
                vram_budget_mb = 8000

        self._pool_mb = pool_mb
        self._vram_budget_mb = vram_budget_mb
        self._prefetch_window = prefetch_window
        self._telemetry = telemetry
        self._block_threshold_mb = block_threshold_mb
        self._slab_mb = 700

        # Defer PinnedPool allocation until first runtime is created.
        # Allocating 7+ GB of pinned (locked) RAM at startup leaves too
        # little headroom for model loading (state dict + key conversion
        # can temporarily need 3x model size in RAM).
        self._pool: PinnedPool | None = None
        self._runtimes: dict[int, object] = {}
        self._lock = threading.Lock()

        # Graph-level scheduling
        self.current_schedule: GraphSchedule | None = None

        logger.info(
            "StagehandCoordinator init (pool deferred): vram_budget=%dMB, prefetch=%d, block_threshold=%dMB",
            vram_budget_mb, prefetch_window, block_threshold_mb,
        )

    def _ensure_pool(self) -> PinnedPool:
        """Lazily allocate the PinnedPool on first use."""
        if self._pool is None:
            pool_mb = max(self._slab_mb, (self._pool_mb // self._slab_mb) * self._slab_mb)
            self._pool = PinnedPool(total_mb=pool_mb, slab_mb=self._slab_mb)
            logger.info("PinnedPool allocated: %d MB (%d slabs x %d MB)",
                        pool_mb, pool_mb // self._slab_mb, self._slab_mb)
        return self._pool

    def get_or_create_runtime(self, model: torch.nn.Module) -> object | None:
        """Create a StagehandRuntime for qualifying models. Returns None if not eligible."""
        model_id = id(model)
        with self._lock:
            if model_id in self._runtimes:
                return self._runtimes[model_id]

        model_mb = _model_size_mb(model)
        if model_mb < self._block_threshold_mb:
            logger.debug(
                "Stagehand: %s too small (%dMB < %dMB threshold), skipping",
                model.__class__.__name__, model_mb, self._block_threshold_mb,
            )
            return None

        arch_name, block_pattern, total_blocks = _detect_architecture(model)
        if block_pattern is None:
            return None

        config = StagehandConfig(
            stagehand_enabled=True,
            pinned_pool_mb=self._pool_mb,
            vram_high_watermark_mb=self._vram_budget_mb,
            vram_low_watermark_mb=int(self._vram_budget_mb * 0.8),
            prefetch_window_blocks=self._prefetch_window,
            telemetry_enabled=self._telemetry,
        )

        pool = self._ensure_pool()
        runtime = StagehandRuntime(
            model=model,
            config=config,
            block_pattern=block_pattern,
            group="transformer",
            dtype=torch.bfloat16,
            inference_mode=True,
            pool=pool,
        )

        with self._lock:
            self._runtimes[model_id] = runtime

        logger.info(
            "Stagehand: Created runtime for %s (arch=%s, %d blocks)",
            model.__class__.__name__, arch_name, total_blocks,
        )
        return runtime

    def managed_forward_ctx(self, model: torch.nn.Module):
        """Return managed_forward() context manager for the model's runtime, or None."""
        model_id = id(model)
        with self._lock:
            runtime = self._runtimes.get(model_id)
        if runtime is not None:
            return runtime.managed_forward()
        return None

    # Graph-level model scheduling

    def set_schedule(self, schedule: GraphSchedule | None):
        """Called before execution loop starts."""
        self.current_schedule = schedule
        if schedule is not None:
            logger.info(
                "Graph planner: %d models tracked, %d nodes with plans",
                len(schedule.model_lifetimes),
                len(schedule.node_plans),
            )

    def prepare_for_node(self, node_id: str):
        """Called before each node executes. Prefetch models that will be needed."""
        if self.current_schedule is None or self.current_schedule.was_invalidated():
            return

        plan = self.current_schedule.get_plan_for_node(node_id)
        if plan is None:
            return

        for model_key in plan.prefetch_before:
            logger.info("Graph planner: PREFETCH %s for node %s", model_key, node_id)

    def complete_node(self, node_id: str):
        """Called after node completes. Evict models that won't be needed again."""
        if self.current_schedule is None or self.current_schedule.was_invalidated():
            return

        plan = self.current_schedule.get_plan_for_node(node_id)
        if plan is None:
            return

        for model_key in plan.evict_after:
            logger.info("Graph planner: EVICT %s (last used by node %s)", model_key, node_id)

    def on_block_h2d(self, block_name: str, gpu_tensor: torch.Tensor,
                     cpu_tensor: torch.Tensor, patch_ledger=None) -> torch.Tensor:
        """Called by Stagehand when transferring a block to GPU.

        If the block is dirty in the patch_ledger, apply patches from the
        clean CPU source to the GPU copy.
        """
        if patch_ledger is None:
            return gpu_tensor

        from serenityflow.core.patch_ledger import PatchLedger
        if not isinstance(patch_ledger, PatchLedger):
            return gpu_tensor

        if patch_ledger.block_is_dirty(block_name):
            from serenityflow.memory.patch_integration import apply_patches_to_block
            gpu_tensor = apply_patches_to_block(
                block_name, gpu_tensor, cpu_tensor, patch_ledger,
            )
        return gpu_tensor

    def has_runtime(self, model_id: int) -> bool:
        with self._lock:
            return model_id in self._runtimes

    def get_runtime(self, model_id: int) -> object | None:
        with self._lock:
            return self._runtimes.get(model_id)

    def release_model(self, model_id: int):
        """Shutdown runtime for model, keeping pool alive for reuse."""
        with self._lock:
            runtime = self._runtimes.pop(model_id, None)
        if runtime is not None:
            try:
                returned_pool = runtime.shutdown_keep_pool()
                if returned_pool is not self._pool:
                    logger.warning("Stagehand: returned pool differs from shared pool")
            except Exception:
                logger.warning("Stagehand: Error during runtime shutdown", exc_info=True)
            logger.info("Stagehand: Released runtime for model id %d", model_id)

    def shutdown(self):
        """Clean shutdown of all runtimes and the shared pool."""
        with self._lock:
            runtime_ids = list(self._runtimes.keys())

        for mid in runtime_ids:
            self.release_model(mid)

        if self._pool is not None:
            try:
                self._pool.shutdown()
            except Exception:
                logger.warning("Stagehand: Error during pool shutdown", exc_info=True)
            self._pool = None

        logger.info("StagehandCoordinator shutdown complete")


__all__ = ["StagehandCoordinator", "BLOCK_PATTERNS"]
