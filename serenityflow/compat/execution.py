"""Compatibility shim for execution module.

Minimal PromptExecutor reference.
"""
from __future__ import annotations


class PromptExecutor:
    def __init__(self, server=None):
        self.server = server
        self.outputs = {}
        self.old_prompt = {}
        self.status_messages = []

    def execute(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
        pass

    def add_message(self, event, data, broadcast):
        self.status_messages.append((event, data))


def validate_prompt(prompt):
    return True, None, list(prompt.keys()), {}
