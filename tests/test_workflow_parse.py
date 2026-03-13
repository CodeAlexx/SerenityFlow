"""Tests for workflow parsing: API format, litegraph format, auto-detection."""
from __future__ import annotations

import json
import os
import pytest

from serenityflow.bridge.workflow import (
    parse_workflow,
    parse_api_format,
    parse_litegraph_format,
)


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


class TestApiFormat:
    def test_parse_basic_flux(self):
        data = load_fixture("basic_flux_t2i.json")
        prompt = parse_api_format(data)
        assert "1" in prompt
        assert prompt["1"]["class_type"] == "UNETLoader"

    def test_validates_link_targets(self):
        with pytest.raises(ValueError, match="non-existent"):
            parse_api_format({
                "1": {"class_type": "Foo", "inputs": {"x": ["999", 0]}},
            })

    def test_missing_inputs_gets_default(self):
        prompt = parse_api_format({
            "1": {"class_type": "Foo"},
        })
        assert prompt["1"]["inputs"] == {}

    def test_missing_class_type_raises(self):
        with pytest.raises(ValueError, match="missing 'class_type'"):
            parse_api_format({"1": {"inputs": {}}})


class TestLitegraphFormat:
    def test_parse_basic_litegraph(self):
        data = load_fixture("basic_litegraph.json")
        prompt = parse_litegraph_format(data)
        assert "1" in prompt
        assert prompt["1"]["class_type"] == "CheckpointLoaderSimple"

    def test_links_converted_correctly(self):
        data = load_fixture("basic_litegraph.json")
        prompt = parse_litegraph_format(data)

        # Node 2 (CLIPTextEncode) should link to node 1 (CheckpointLoader) output 1
        clip_input = prompt["2"]["inputs"].get("clip")
        assert clip_input is not None
        assert clip_input == ["1", 1]

    def test_empty_nodes_raises(self):
        with pytest.raises(ValueError, match="no nodes"):
            parse_litegraph_format({"nodes": [], "links": []})


class TestAutoDetection:
    def test_detects_api_format(self):
        data = load_fixture("basic_flux_t2i.json")
        prompt = parse_workflow(data)
        assert "1" in prompt
        assert prompt["1"]["class_type"] == "UNETLoader"

    def test_detects_litegraph_format(self):
        data = load_fixture("basic_litegraph.json")
        prompt = parse_workflow(data)
        assert "1" in prompt
        assert prompt["1"]["class_type"] == "CheckpointLoaderSimple"
