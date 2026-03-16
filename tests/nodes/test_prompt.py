"""Tests for prompt_utils nodes -- text ops, prompt scheduling, conditioning helpers."""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# TextConcatenate
# ---------------------------------------------------------------------------

class TestTextConcatenate:
    def test_registered(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("TextConcatenate")

    def test_default_separator(self):
        from serenityflow.nodes.prompt_utils import text_concatenate
        (result,) = text_concatenate("hello", "world")
        assert result == "hello, world"

    def test_custom_separator(self):
        from serenityflow.nodes.prompt_utils import text_concatenate
        (result,) = text_concatenate("hello", "world", separator=" | ")
        assert result == "hello | world"

    def test_empty_separator(self):
        from serenityflow.nodes.prompt_utils import text_concatenate
        (result,) = text_concatenate("foo", "bar", separator="")
        assert result == "foobar"


# ---------------------------------------------------------------------------
# TextMultiline
# ---------------------------------------------------------------------------

class TestTextMultiline:
    def test_registered(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("TextMultiline")

    def test_passthrough(self):
        from serenityflow.nodes.prompt_utils import text_multiline
        text = "line1\nline2\nline3"
        (result,) = text_multiline(text)
        assert result == text


# ---------------------------------------------------------------------------
# ShowText
# ---------------------------------------------------------------------------

class TestShowText:
    def test_registered(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("ShowText")

    def test_passthrough(self):
        from serenityflow.nodes.prompt_utils import show_text
        (result,) = show_text("preview me")
        assert result == "preview me"

    def test_returns_string_type(self):
        from serenityflow.nodes.registry import registry
        node = registry.get("ShowText")
        assert node.return_types == ("STRING",)


# ---------------------------------------------------------------------------
# PromptSchedule
# ---------------------------------------------------------------------------

class TestPromptSchedule:
    def test_registered(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("PromptSchedule")

    def test_parse_schedule(self):
        from serenityflow.nodes.prompt_utils import _parse_schedule
        text = "0: a cat\n10: a dog\n20: a bird"
        entries = _parse_schedule(text)
        assert len(entries) == 3
        assert entries[0] == (0, "a cat")
        assert entries[1] == (10, "a dog")
        assert entries[2] == (20, "a bird")

    def test_parse_schedule_unordered(self):
        from serenityflow.nodes.prompt_utils import _parse_schedule
        text = "20: third\n0: first\n10: second"
        entries = _parse_schedule(text)
        assert entries[0][0] == 0
        assert entries[1][0] == 10
        assert entries[2][0] == 20

    def test_parse_schedule_skips_bad_lines(self):
        from serenityflow.nodes.prompt_utils import _parse_schedule
        text = "0: good\nbadline\n10: also good\n\n"
        entries = _parse_schedule(text)
        assert len(entries) == 2

    def test_conditioning_structure(self):
        """Verify prompt_schedule produces conditioning with timestep ranges."""
        from serenityflow.nodes.prompt_utils import prompt_schedule

        mock_cond = [{"cross_attn": "tensor_placeholder", "pooled_output": "pool"}]

        def fake_encode(clip, text):
            return [{"cross_attn": f"encoded:{text}", "pooled_output": "pool"}]

        with patch("serenityflow.bridge.serenity_api.encode_text", fake_encode):
            schedule_text = "0: a cat\n500: a dog"
            (result,) = prompt_schedule(schedule_text, clip="dummy_clip")

        assert isinstance(result, list)
        assert len(result) == 2
        # First entry: step 0 to 500
        assert result[0]["timestep_start"] == 0.0
        assert result[0]["timestep_end"] == 0.5
        assert "cat" in result[0]["cross_attn"]
        # Second entry: step 500 to 1000
        assert result[1]["timestep_start"] == 0.5
        assert result[1]["timestep_end"] == 1.0
        assert "dog" in result[1]["cross_attn"]

    def test_single_entry_fallback(self):
        """Single entry covers the entire range."""
        from serenityflow.nodes.prompt_utils import prompt_schedule

        def fake_encode(clip, text):
            return [{"cross_attn": f"encoded:{text}"}]

        with patch("serenityflow.bridge.serenity_api.encode_text", fake_encode):
            (result,) = prompt_schedule("0: only prompt", clip="dummy")

        assert len(result) == 1
        assert result[0]["timestep_start"] == 0.0
        assert result[0]["timestep_end"] == 1.0

    def test_empty_schedule_fallback(self):
        """Empty/invalid schedule falls back to encoding full text."""
        from serenityflow.nodes.prompt_utils import prompt_schedule

        def fake_encode(clip, text):
            return [{"cross_attn": f"encoded:{text}"}]

        with patch("serenityflow.bridge.serenity_api.encode_text", fake_encode):
            (result,) = prompt_schedule("just plain text, no schedule", clip="dummy")

        assert isinstance(result, list)
        assert len(result) == 1
        assert "just plain text" in result[0]["cross_attn"]


# ---------------------------------------------------------------------------
# ConditioningBatchCompose
# ---------------------------------------------------------------------------

class TestConditioningBatchCompose:
    def test_registered(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("ConditioningBatchCompose")

    def test_compose(self):
        from serenityflow.nodes.prompt_utils import conditioning_batch_compose
        c1 = [{"cross_attn": "a"}, {"cross_attn": "b"}]
        c2 = [{"cross_attn": "c"}]
        (result,) = conditioning_batch_compose(c1, c2)
        assert len(result) == 3
        assert result[0]["batch_index"] == 0
        assert result[1]["batch_index"] == 1
        assert result[2]["batch_index"] == 2
        assert result[2]["cross_attn"] == "c"
