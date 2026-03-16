"""LanPaint state types."""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch


class LangevinState(NamedTuple):
    """State carried between Langevin dynamics iterations."""
    v: Optional[torch.Tensor]
    C: Optional[torch.Tensor]
    x0: Optional[torch.Tensor]


__all__ = ["LangevinState"]
