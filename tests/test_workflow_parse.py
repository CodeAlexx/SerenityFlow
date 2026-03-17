"""Tests for workflow parsing: API format, litegraph format, auto-detection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from serenityflow.bridge.workflow import (
    parse_workflow,
    parse_api_format,
    parse_litegraph_format,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
WORKFLOWS = Path(__file__).resolve().parents[1] / "serenityflow" / "workflows"


def load_fixture(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


def load_workflow(name):
    with open(WORKFLOWS / name) as f:
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

    def test_parse_ltx23_t2v_workflow(self):
        data = load_workflow("ltx23_t2v.json")
        prompt = parse_workflow(data)

        assert prompt["1"]["class_type"] == "LTXVLoader"
        assert prompt["1"]["inputs"]["backend"] == "legacy_stagehand"
        assert prompt["2"]["class_type"] == "LTXVSampler"
        assert prompt["2"]["inputs"]["ltxv_model"] == ["1", 0]
        assert "guide_image" not in prompt["2"]["inputs"]
        assert "audio" not in prompt["2"]["inputs"]
        assert prompt["3"]["inputs"]["video"] == ["2", 1]

    def test_parse_ltx23_i2v_workflow(self):
        data = load_workflow("ltx23_i2v.json")
        prompt = parse_workflow(data)

        assert prompt["1"]["class_type"] == "LTXVLoader"
        assert prompt["1"]["inputs"]["backend"] == "legacy_stagehand"
        assert prompt["2"]["class_type"] == "LoadImage"
        assert prompt["3"]["class_type"] == "LTXVSampler"
        assert prompt["3"]["inputs"]["ltxv_model"] == ["1", 0]
        assert prompt["3"]["inputs"]["guide_image"] == ["2", 0]
        assert prompt["3"]["inputs"]["guide_frame_idx"] == 0
        assert "audio" not in prompt["3"]["inputs"]
        assert prompt["4"]["inputs"]["video"] == ["3", 1]

    def test_parse_ltx23_ia2v_workflow(self):
        data = load_workflow("ltx23_ia2v.json")
        prompt = parse_workflow(data)

        assert prompt["1"]["class_type"] == "LTXVLoader"
        assert prompt["1"]["inputs"]["backend"] == "legacy_stagehand"
        assert prompt["2"]["class_type"] == "LoadImage"
        assert prompt["3"]["class_type"] == "LoadAudio"
        assert prompt["4"]["class_type"] == "LTXVSampler"
        assert prompt["4"]["inputs"]["ltxv_model"] == ["1", 0]
        assert prompt["4"]["inputs"]["guide_image"] == ["2", 0]
        assert prompt["4"]["inputs"]["audio"] == ["3", 0]
        assert prompt["4"]["inputs"]["audio_start_time"] == 0.0
        assert prompt["5"]["inputs"]["video"] == ["4", 1]
        assert prompt["6"]["inputs"]["audio"] == ["4", 2]


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

    def test_detects_ltx23_t2v_workflow(self):
        prompt = parse_workflow(load_workflow("ltx23_t2v.json"))

        assert prompt["1"]["class_type"] == "LTXVLoader"
        assert prompt["1"]["inputs"]["backend"] == "legacy_stagehand"
        assert prompt["2"]["class_type"] == "LTXVSampler"
        assert prompt["2"]["inputs"]["ltxv_model"] == ["1", 0]
        assert "guide_image" not in prompt["2"]["inputs"]
        assert "audio" not in prompt["2"]["inputs"]
        assert prompt["3"]["class_type"] == "SaveVideo"
        assert prompt["3"]["inputs"]["video"] == ["2", 1]

    def test_detects_ltx23_i2v_workflow(self):
        prompt = parse_workflow(load_workflow("ltx23_i2v.json"))

        assert prompt["1"]["class_type"] == "LTXVLoader"
        assert prompt["1"]["inputs"]["backend"] == "legacy_stagehand"
        assert prompt["2"]["class_type"] == "LoadImage"
        assert prompt["3"]["class_type"] == "LTXVSampler"
        assert prompt["3"]["inputs"]["guide_image"] == ["2", 0]
        assert prompt["3"]["inputs"]["guide_strength"] == 1.0
        assert prompt["3"]["inputs"]["guide_frame_idx"] == 0
        assert "audio" not in prompt["3"]["inputs"]
        assert prompt["4"]["class_type"] == "SaveVideo"
        assert prompt["4"]["inputs"]["video"] == ["3", 1]

    def test_detects_ltx23_ia2v_workflow(self):
        prompt = parse_workflow(load_workflow("ltx23_ia2v.json"))

        assert prompt["1"]["class_type"] == "LTXVLoader"
        assert prompt["1"]["inputs"]["backend"] == "legacy_stagehand"
        assert prompt["2"]["class_type"] == "LoadImage"
        assert prompt["3"]["class_type"] == "LoadAudio"
        assert prompt["4"]["class_type"] == "LTXVSampler"
        assert prompt["4"]["inputs"]["guide_image"] == ["2", 0]
        assert prompt["4"]["inputs"]["audio"] == ["3", 0]
        assert prompt["4"]["inputs"]["audio_start_time"] == 0.0
        assert prompt["4"]["inputs"]["audio_duration"] == 4.0
        assert prompt["5"]["class_type"] == "SaveVideo"
        assert prompt["5"]["inputs"]["video"] == ["4", 1]
        assert prompt["6"]["class_type"] == "SaveAudioOpus"
        assert prompt["6"]["inputs"]["audio"] == ["4", 2]
