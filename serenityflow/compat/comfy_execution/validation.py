"""Compatibility shim for comfy_execution.validation."""
from __future__ import annotations


def validate_prompt(prompt):
    """Validate a prompt dict. Returns (valid, error, good_outputs, node_errors)."""
    return True, None, list(prompt.keys()), {}
