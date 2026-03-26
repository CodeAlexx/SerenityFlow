"""Compatibility alias for pinned pool types.

Primary implementation lives in :mod:`stagehand.pool`.
"""

from .pool import PinnedPool, PinnedSlab

__all__ = ["PinnedPool", "PinnedSlab"]
