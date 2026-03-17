from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Gemma3Config

from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from serenityflow.bridge import model_paths
from serenityflow.bridge.serenity_api import (
    _build_ltxv_stage2_ledger,
    _ltx_gemma_rope_profiles,
    _ltx_prepare_gemma_token_pairs,
    _materialize_ltxv_audio_path,
    _materialize_ltxv_image_conditionings,
    _resolve_ltxv_gemma_root,
    _should_use_official_ltx_backend,
)


def test_ltx_gemma_rope_profiles_handle_nested_transformers_config():
    config = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict()).text_config

    local_base, full_cfg, full_theta = _ltx_gemma_rope_profiles(config)

    assert local_base == 10000
    assert full_cfg["rope_type"] == "linear"
    assert full_theta == 1000000


def test_resolve_ltxv_gemma_root_prefers_standalone_dir(tmp_path, monkeypatch):
    text_encoders = tmp_path / "models" / "text_encoders"
    gptq_dir = text_encoders / "gemma-3-12b-it-GPTQ-4b"
    standalone_dir = text_encoders / "gemma-3-12b-it-standalone"
    for directory in (gptq_dir, standalone_dir):
        directory.mkdir(parents=True)
        (directory / "tokenizer.model").write_bytes(b"tok")
        (directory / "preprocessor_config.json").write_text("{}")

    monkeypatch.setattr(model_paths, "_UNIFIED_BASE", str(tmp_path / "models"))
    monkeypatch.setattr(model_paths, "_WELL_KNOWN_MODEL_DIRS", [])
    monkeypatch.setattr(model_paths, "_instance", model_paths.ModelPaths(str(tmp_path)))

    assert _resolve_ltxv_gemma_root("gemma-3-12b-it") == str(standalone_dir)


def test_should_use_official_ltx_backend_only_for_explicit_official_backend():
    guide_image = torch.zeros(1, 8, 8, 3)
    audio = {"path": "/tmp/fake.wav"}

    assert _should_use_official_ltx_backend(SimpleNamespace(backend="official")) is True
    assert _should_use_official_ltx_backend(
        SimpleNamespace(backend="auto"),
        guide_image=guide_image,
        audio=audio,
    ) is False
    assert _should_use_official_ltx_backend(
        SimpleNamespace(backend="auto"),
        guide_image=guide_image,
        audio=None,
    ) is False
    assert _should_use_official_ltx_backend(
        SimpleNamespace(backend="auto"),
        guide_image=None,
        audio=audio,
    ) is False
    assert _should_use_official_ltx_backend(
        SimpleNamespace(backend="stagehand"),
        guide_image=guide_image,
        audio=audio,
    ) is False
    assert _should_use_official_ltx_backend(
        SimpleNamespace(backend="legacy_stagehand"),
        guide_image=guide_image,
        audio=audio,
    ) is False


def test_build_ltxv_stage2_ledger_adds_distilled_lora(monkeypatch):
    calls = []

    class DummyLedger:
        def with_loras(self, loras):
            calls.append(loras)
            return "stage2-ledger"

    monkeypatch.setattr(
        "serenityflow.bridge.serenity_api._build_ltxv_loras",
        lambda paths, strengths: [("distilled", paths, strengths)],
    )

    result = _build_ltxv_stage2_ledger(
        SimpleNamespace(
            model_ledger=DummyLedger(),
            distilled_lora_path="/tmp/ltx-distilled.safetensors",
        )
    )

    assert result == "stage2-ledger"
    assert calls == [(("distilled", ("/tmp/ltx-distilled.safetensors",), (1.0,)),)]


def test_prepare_gemma_token_pairs_pads_to_connector_multiple():
    fake_tokenizer = SimpleNamespace(
        tokenize_with_weights=lambda text: {"gemma": [(11, 0), (12, 0), (13, 1), (14, 1), (15, 1)]},
        tokenizer=SimpleNamespace(pad_token_id=99),
    )
    fake_text_encoder = SimpleNamespace(
        tokenizer=fake_tokenizer,
        embeddings_processor=SimpleNamespace(
            video_connector=SimpleNamespace(num_learnable_registers=4),
        ),
    )

    token_pairs = _ltx_prepare_gemma_token_pairs(fake_text_encoder, "hello")

    assert len(token_pairs) == 4
    assert token_pairs[:1] == [(99, 0)]
    assert token_pairs[1:] == [(13, 1), (14, 1), (15, 1)]


def test_materialize_ltxv_image_conditionings_clamps_frame_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    guide_image = torch.ones(16, 16, 3)
    items = _materialize_ltxv_image_conditionings(
        guide_image=guide_image,
        guide_frame_idx=-5,
        guide_strength=0.75,
    )

    assert len(items) == 1
    assert Path(items[0].path).exists()
    assert items[0].frame_idx == 0
    assert items[0].strength == 0.75


def test_materialize_ltxv_image_conditionings_none_returns_empty():
    assert _materialize_ltxv_image_conditionings(None, 0, 1.0) == []


def test_materialize_ltxv_audio_path_writes_waveform_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    waveform = torch.zeros(2, 320, dtype=torch.float32)
    audio_path = _materialize_ltxv_audio_path(
        {
            "waveform": waveform,
            "sample_rate": 16_000,
        }
    )

    assert audio_path is not None
    assert Path(audio_path).exists()
    assert audio_path.endswith(".wav")


def test_materialize_ltxv_audio_path_passthrough_for_loadaudio_dict(tmp_path, monkeypatch):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")

    def fail_write(*args, **kwargs):
        raise AssertionError("waveform writer should not be used for path-backed audio")

    monkeypatch.setattr("serenityflow.bridge.serenity_api._write_audio_waveform_to_wav", fail_write)

    assert _materialize_ltxv_audio_path(
        {
            "path": str(clip),
            "waveform": None,
            "sample_rate": None,
        }
    ) == str(clip)
