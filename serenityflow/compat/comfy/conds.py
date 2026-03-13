"""Compatibility shim for comfy.conds."""
from __future__ import annotations

import torch


class CONDRegular:
    def __init__(self, cond):
        self.cond = cond

    def can_concat_to(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat_to(self, others):
        return torch.cat([self.cond] + [o.cond for o in others])


class CONDNoiseShape(CONDRegular):
    pass


class CONDCrossAttn(CONDRegular):
    def can_concat_to(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:
                return False
        return True

    def concat_to(self, others):
        conds = [self.cond] + [o.cond for o in others]
        max_len = max(c.shape[1] for c in conds)
        result = []
        for c in conds:
            if c.shape[1] < max_len:
                c = torch.nn.functional.pad(c, (0, 0, 0, max_len - c.shape[1]))
            result.append(c)
        return torch.cat(result)


class CONDConstant:
    def __init__(self, cond):
        self.cond = cond

    def can_concat_to(self, other):
        return self.cond == other.cond

    def concat_to(self, others):
        return self.cond
