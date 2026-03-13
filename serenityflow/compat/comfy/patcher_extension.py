"""Compatibility shim for comfy.patcher_extension.

Callback/wrapper registration constants and helpers.
"""
from __future__ import annotations

import copy


# Callback types
class CallbacksMP:
    ON_CLONE = "on_clone"
    ON_LOAD = "on_load"
    ON_CLEANUP = "on_cleanup"
    ON_PRE_RUN = "on_pre_run"
    ON_PREPARE_STATE = "on_prepare_state"
    ON_APPLY_HOOKS = "on_apply_hooks"
    ON_REGISTER_HOOKS = "on_register_hooks"
    ON_INJECT_MODEL = "on_inject_model"
    ON_EJECT_MODEL = "on_eject_model"
    ON_WEIGHT_LOAD = "on_weight_load"


# Wrapper types
class WrappersMP:
    OUTER_SAMPLE = "outer_sample"
    SAMPLER_SAMPLE = "sampler_sample"
    CALC_COND_BATCH = "calc_cond_batch"
    APPLY_MODEL = "apply_model"
    DIFFUSION_MODEL = "diffusion_model"


class PatcherInjection:
    def __init__(self, inject=None, eject=None):
        self.inject = inject
        self.eject = eject


class WrapperExecutor:
    """Executes a chain of wrappers around a function."""

    def __init__(self, original, wrappers, *args, **kwargs):
        self.original = original
        self.wrappers = list(wrappers)
        self.args = args
        self.kwargs = kwargs
        self._idx = 0

    def execute(self, *args, **kwargs):
        if self._idx < len(self.wrappers):
            wrapper = self.wrappers[self._idx]
            self._idx += 1
            return wrapper(self, *args, **kwargs)
        return self.original(*args, **kwargs)

    @classmethod
    def new_executor(cls, original, wrappers, *args, **kwargs):
        return cls(original, wrappers, *args, **kwargs)


def copy_nested_dicts(d):
    """Deep copy a dict of dicts."""
    if not isinstance(d, dict):
        return d
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = copy_nested_dicts(v)
        elif isinstance(v, list):
            result[k] = v[:]
        else:
            result[k] = v
    return result


def merge_nested_dicts(base, override, copy=True):
    if copy:
        base = copy_nested_dicts(base)
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            merge_nested_dicts(base[k], v, copy=False)
        elif k in base and isinstance(base[k], list) and isinstance(v, list):
            base[k] = base[k] + v
        else:
            base[k] = v
    return base
