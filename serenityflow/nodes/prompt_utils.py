"""Prompt utility nodes -- text manipulation, prompt scheduling, conditioning helpers."""
from __future__ import annotations

import re

from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# String / text utilities
# ---------------------------------------------------------------------------

@registry.register(
    "TextConcatenate",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {
        "text1": ("STRING",),
        "text2": ("STRING",),
    },
    "optional": {
        "separator": ("STRING",),
    }},
)
def text_concatenate(text1, text2, separator=", "):
    """Join two strings with a configurable separator."""
    return (f"{text1}{separator}{text2}",)


@registry.register(
    "TextMultiline",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {"text": ("STRING",)}},
)
def text_multiline(text):
    """Multiline text input -- pass-through."""
    return (text,)


@registry.register(
    "ShowText",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {"text": ("STRING",)}},
)
def show_text(text):
    """Display text as a preview node.  Returns the input unchanged."""
    return (text,)


# ---------------------------------------------------------------------------
# Prompt scheduling
# ---------------------------------------------------------------------------

def _parse_schedule(text: str) -> list[tuple[int, str]]:
    """Parse a prompt schedule string into (step, prompt) pairs.

    Expected format (one entry per line):
        0: a photo of a cat
        10: a painting of a dog
        20: an abstract landscape

    Lines that don't match ``<int>: <text>`` are silently skipped.
    """
    entries: list[tuple[int, str]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s*:\s*(.+)$", line)
        if m:
            entries.append((int(m.group(1)), m.group(2).strip()))
    entries.sort(key=lambda x: x[0])
    return entries


@registry.register(
    "PromptSchedule",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "text": ("STRING",),
        "clip": ("CLIP",),
    }},
)
def prompt_schedule(text, clip):
    """Encode different prompts for different sampling step ranges.

    Format::

        0: prompt_a
        10: prompt_b
        20: prompt_c

    Each prompt is active from its step number until the next entry.
    Timestep ranges are normalised to [0, 1] assuming 1000 total steps by
    default -- the sampler will map to actual sigma ranges at runtime.
    """
    from serenityflow.bridge.serenity_api import encode_text

    entries = _parse_schedule(text)
    if not entries:
        # Fallback: treat entire text as a single prompt
        return (encode_text(clip, text),)

    # Determine the max step for normalisation
    max_step = max(step for step, _ in entries)
    # Use at least 1000 to avoid division by zero / tiny ranges
    total_steps = max(max_step + 1, 1000)

    all_conds: list[dict] = []
    for i, (step, prompt) in enumerate(entries):
        cond_list = encode_text(clip, prompt)
        # Determine range end
        if i + 1 < len(entries):
            end_step = entries[i + 1][0]
        else:
            end_step = total_steps

        start_frac = step / total_steps
        end_frac = end_step / total_steps

        for c in cond_list:
            n = dict(c)
            n["timestep_start"] = start_frac
            n["timestep_end"] = end_frac
            all_conds.append(n)

    return (all_conds,)


# ---------------------------------------------------------------------------
# Conditioning batch compose
# ---------------------------------------------------------------------------

@registry.register(
    "ConditioningBatchCompose",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning_1": ("CONDITIONING",),
        "conditioning_2": ("CONDITIONING",),
    }},
)
def conditioning_batch_compose(conditioning_1, conditioning_2):
    """Compose different conditionings for batch items.

    Combines two conditioning lists so that each element targets a
    different batch item during sampling.
    """
    out: list[dict] = []
    for i, c in enumerate(conditioning_1):
        n = dict(c)
        n["batch_index"] = i
        out.append(n)
    offset = len(conditioning_1)
    for i, c in enumerate(conditioning_2):
        n = dict(c)
        n["batch_index"] = offset + i
        out.append(n)
    return (out,)
