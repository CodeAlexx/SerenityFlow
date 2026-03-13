"""Compatibility shim for comfy.hooks.

Provides HookGroup, HookKeyframeGroup, HookKeyframe, and conditioning helpers.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class EnumHookMode(Enum):
    MinVram = 0
    MaxSpeed = 1


class EnumHookType(Enum):
    Weight = 0
    Patch = 1
    ObjectPatch = 2
    AddModels = 3
    Callbacks = 4
    Wrappers = 5
    SetInjections = 6


class EnumWeightTarget(Enum):
    Model = 0
    Clip = 1


class HookKeyframe:
    def __init__(self, strength=1.0, start_percent=0.0, guarantee_steps=1):
        self.strength = strength
        self.start_percent = start_percent
        self.guarantee_steps = guarantee_steps


class HookKeyframeGroup:
    def __init__(self):
        self.keyframes: list[HookKeyframe] = []

    def add(self, keyframe: HookKeyframe):
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.start_percent)
        return self

    def clone(self):
        n = HookKeyframeGroup()
        n.keyframes = self.keyframes[:]
        return n


class Hook:
    def __init__(self, hook_type=EnumHookType.Weight, hook_ref=None,
                 hook_id=None, hook_keyframes=None):
        self.hook_type = hook_type
        self.hook_ref = hook_ref
        self.hook_id = hook_id or id(self)
        self.hook_keyframes = hook_keyframes or HookKeyframeGroup()
        self.strength = 1.0
        self.hook_mode = EnumHookMode.MinVram

    def clone(self, subtype=None):
        import copy
        return copy.copy(self)


class WeightHook(Hook):
    def __init__(self, strength_model=1.0, strength_clip=1.0, **kwargs):
        super().__init__(hook_type=EnumHookType.Weight, **kwargs)
        self.strength_model = strength_model
        self.strength_clip = strength_clip


class HookGroup:
    def __init__(self):
        self.hooks: list[Hook] = []

    def add(self, hook: Hook):
        self.hooks.append(hook)
        return self

    def clone(self):
        n = HookGroup()
        n.hooks = self.hooks[:]
        return n

    def clone_and_combine(self, other: HookGroup | None):
        n = self.clone()
        if other is not None:
            for h in other.hooks:
                n.hooks.append(h)
        return n

    def set_keyframes_on_hooks(self, hook_kf: HookKeyframeGroup):
        for h in self.hooks:
            h.hook_keyframes = hook_kf

    def get_type(self, hook_type: EnumHookType):
        return [h for h in self.hooks if h.hook_type == hook_type]

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)


def set_hooks_for_conditioning(cond, hooks: HookGroup | None):
    if hooks is None:
        return cond
    result = []
    for c in cond:
        n = [c[0], c[1].copy()]
        n[1]["hooks"] = hooks
        result.append(n)
    return result


def set_timesteps_for_conditioning(cond, start_percent=0.0, end_percent=1.0):
    result = []
    for c in cond:
        n = [c[0], c[1].copy()]
        n[1]["start_percent"] = start_percent
        n[1]["end_percent"] = end_percent
        result.append(n)
    return result


def set_conds_props_and_combine(conds, new_conds, strength=1.0,
                                 set_cond_area="default", hooks=None,
                                 timestep_range=None):
    if timestep_range is not None:
        new_conds = set_timesteps_for_conditioning(
            new_conds, timestep_range[0], timestep_range[1],
        )
    if hooks is not None:
        new_conds = set_hooks_for_conditioning(new_conds, hooks)
    if strength != 1.0:
        for c in new_conds:
            c[1]["strength"] = strength
    if conds is None:
        return new_conds
    return conds + new_conds


def conditioning_set_values(cond, values):
    result = []
    for c in cond:
        n = [c[0], c[1].copy()]
        n[1].update(values)
        result.append(n)
    return result
