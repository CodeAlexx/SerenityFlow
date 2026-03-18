from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from collections import namedtuple
from types import ModuleType

import torch
from transformers import Gemma3Config

from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from serenityflow.bridge import model_paths
from serenityflow.bridge.serenity_api import (
    _build_ltxv_stage2_ledger,
    _desktop_fast_ltx_checkpoint_candidates,
    _default_ltxv_distilled_lora_candidates,
    _ltx_gemma_weight_bytes,
    _ltx_gemma_rope_profiles,
    _maybe_prefer_desktop_fast_ltx_checkpoint,
    _prepare_ltx_scaled_fp8_transformer_for_runtime,
    _ltx_scaled_fp8_runtime_backend,
    _ltx_text_encoder_device_candidates,
    _ltx_prepare_gemma_token_pairs,
    _materialize_ltxv_audio_path,
    _materialize_ltxv_image_conditionings,
    _wrap_official_ltx_ledger_transformer_cpu_stage,
    _wrap_ltx_ledger_text_encoder_cpu,
    _resolve_ltxv_asset,
    _resolve_ltxv_asset_from_hf_cache,
    _resolve_ltxv_gemma_root,
    _should_try_cuda_for_full_ltx_text_encoder,
    _should_use_official_ltx_backend,
)


def _install_fake_eriquant_backend(
    monkeypatch,
    *,
    is_available=lambda: True,
    quantize_model_eriquant=None,
):
    serenity_mod = ModuleType("serenity")
    inference_mod = ModuleType("serenity.inference")
    quantization_mod = ModuleType("serenity.inference.quantization")
    eriquant_mod = ModuleType("serenity.inference.quantization.eriquant_backend")

    eriquant_mod.is_available = is_available
    eriquant_mod.quantize_model_eriquant = quantize_model_eriquant or (lambda model, **kwargs: None)

    serenity_mod.inference = inference_mod
    inference_mod.quantization = quantization_mod
    quantization_mod.eriquant_backend = eriquant_mod

    monkeypatch.setitem(sys.modules, "serenity", serenity_mod)
    monkeypatch.setitem(sys.modules, "serenity.inference", inference_mod)
    monkeypatch.setitem(sys.modules, "serenity.inference.quantization", quantization_mod)
    monkeypatch.setitem(sys.modules, "serenity.inference.quantization.eriquant_backend", eriquant_mod)

    return eriquant_mod


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


def test_resolve_ltxv_gemma_root_honors_explicit_request(tmp_path, monkeypatch):
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

    assert _resolve_ltxv_gemma_root(str(gptq_dir)) == str(gptq_dir)


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


def test_default_ltxv_distilled_lora_candidates_require_23_lora_for_23_checkpoints():
    assert _default_ltxv_distilled_lora_candidates("ltx-2.3-22b-dev-fp8.safetensors") == (
        "ltx-2.3-22b-distilled-lora-384.safetensors",
    )


def test_default_ltxv_distilled_lora_candidates_keep_19b_for_legacy_checkpoints():
    assert _default_ltxv_distilled_lora_candidates("ltx-2-19b-dev.safetensors") == (
        "ltx-2-19b-distilled-lora-384.safetensors",
    )


def test_ltx_text_encoder_device_candidates_prefer_cuda_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert _ltx_text_encoder_device_candidates(torch.device("cuda")) == (
        torch.device("cuda"),
        torch.device("cpu"),
    )


def test_ltx_text_encoder_device_candidates_use_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _ltx_text_encoder_device_candidates(torch.device("cuda")) == (torch.device("cpu"),)


def test_ltx_gemma_weight_bytes_sums_model_shards_and_caches(tmp_path):
    gemma_root = tmp_path / "gemma"
    gemma_root.mkdir()
    (gemma_root / "model-00001-of-00002.safetensors").write_bytes(b"a" * 10)
    (gemma_root / "model-00002-of-00002.safetensors").write_bytes(b"b" * 7)
    _ltx_gemma_weight_bytes.cache_clear()

    assert _ltx_gemma_weight_bytes(str(gemma_root)) == 17
    assert _ltx_gemma_weight_bytes(str(gemma_root)) == 17


def test_should_try_cuda_for_full_ltx_text_encoder_skips_when_weights_do_not_fit(tmp_path, monkeypatch):
    gemma_root = tmp_path / "gemma"
    gemma_root.mkdir()
    (gemma_root / "model.safetensors").write_bytes(b"x" * 32)
    _ltx_gemma_weight_bytes.cache_clear()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (16, 64))

    ledger = SimpleNamespace(gemma_root_path=str(gemma_root))
    assert _should_try_cuda_for_full_ltx_text_encoder(ledger, torch.device("cuda")) is False


def test_should_try_cuda_for_full_ltx_text_encoder_allows_when_weights_fit(tmp_path, monkeypatch):
    gemma_root = tmp_path / "gemma"
    gemma_root.mkdir()
    (gemma_root / "model.safetensors").write_bytes(b"x" * 16)
    _ltx_gemma_weight_bytes.cache_clear()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2 * (1024**3), 0))

    ledger = SimpleNamespace(gemma_root_path=str(gemma_root))
    assert _should_try_cuda_for_full_ltx_text_encoder(ledger, torch.device("cuda")) is True


def test_wrap_ltx_ledger_text_encoder_cpu_preserves_original_factory(monkeypatch):
    factory_calls = []
    encode_calls = []
    DummyOutput = namedtuple("DummyOutput", ["video_encoding", "audio_encoding", "attention_mask"])

    class DummyTextEncoder:
        def precompute(self, text, padding_side="left"):
            return text, padding_side

        def __call__(self, text, padding_side="left"):
            encode_calls.append((text, padding_side))
            return DummyOutput(torch.ones(1), None, torch.ones(1))

    def original_text_encoder():
        factory_calls.append("original")
        return DummyTextEncoder()

    ledger = SimpleNamespace(
        device=torch.device("cpu"),
        gemma_root_path="",
        text_encoder=original_text_encoder,
    )
    model = SimpleNamespace(
        _cached_text_encoder=None,
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        "serenityflow.bridge.serenity_api._ltx_text_encoder_device_candidates",
        lambda _device: (torch.device("cpu"),),
    )

    _wrap_ltx_ledger_text_encoder_cpu(ledger, model)
    wrapped = ledger.text_encoder()

    result = wrapped("hello")

    assert torch.equal(result.video_encoding, torch.ones(1))
    assert result.audio_encoding is None
    assert torch.equal(result.attention_mask, torch.ones(1))
    assert model._cached_text_encoder is not None
    assert factory_calls == ["original"]
    assert encode_calls == [("hello", "left")]


def test_prepare_ltx_scaled_fp8_transformer_for_runtime_uses_eriquant(monkeypatch):
    transformer = torch.nn.Linear(4, 4, bias=False)
    calls = []

    monkeypatch.setattr(
        "serenityflow.bridge.serenity_api._dequant_scaled_fp8_weights",
        lambda model, checkpoint_path: calls.append(("dequant", type(model).__name__, checkpoint_path)) or 7,
    )
    monkeypatch.setattr(
        "serenityflow.bridge.serenity_api._ltx_scaled_fp8_runtime_backend",
        lambda: "eriquant_fp8",
    )
    _install_fake_eriquant_backend(
        monkeypatch,
        quantize_model_eriquant=lambda model, **kwargs: calls.append(("eriquant", type(model).__name__, kwargs)),
    )

    backend = _prepare_ltx_scaled_fp8_transformer_for_runtime(
        transformer,
        "/tmp/fake.safetensors",
        stage_label="Stage 1",
    )

    assert backend == "eriquant_fp8"
    assert transformer._serenity_scaled_fp8_backend == "eriquant_fp8"
    assert calls[0] == ("dequant", "Linear", "/tmp/fake.safetensors")
    assert calls[1][0] == "eriquant"
    assert calls[1][2]["mode"] == "eriquant_fp8"


def test_ltx_scaled_fp8_runtime_backend_defaults_to_dequant_bf16(monkeypatch):
    monkeypatch.delenv("SERENITY_LTX_SCALED_FP8_BACKEND", raising=False)
    _install_fake_eriquant_backend(monkeypatch, is_available=lambda: True)

    assert _ltx_scaled_fp8_runtime_backend() == "dequant_bf16"


def test_ltx_scaled_fp8_runtime_backend_honors_eriquant_override(monkeypatch):
    monkeypatch.setenv("SERENITY_LTX_SCALED_FP8_BACKEND", "eriquant_fp8")
    _install_fake_eriquant_backend(monkeypatch, is_available=lambda: False)

    assert _ltx_scaled_fp8_runtime_backend() == "eriquant_fp8"


def test_wrap_official_ltx_ledger_transformer_cpu_stage_cpu_stages_and_prepares_runtime(monkeypatch):
    calls = []

    class DummyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))

    def original_transformer():
        calls.append(("factory_device", ledger.device))
        return DummyTransformer()

    ledger = SimpleNamespace(
        device=torch.device("cuda"),
        transformer=original_transformer,
    )
    model = SimpleNamespace(
        device=torch.device("cuda"),
        checkpoint_path="/tmp/fake.safetensors",
        is_scaled_fp8=True,
    )

    prepared = []
    direct_gpu = []

    monkeypatch.setattr(
        "serenityflow.bridge.serenity_api._prepare_ltx_scaled_fp8_transformer_for_runtime",
        lambda transformer, checkpoint_path, **kwargs: prepared.append(
            (type(transformer).__name__, checkpoint_path, kwargs["stage_label"])
        ) or "eriquant_fp8",
    )
    monkeypatch.setattr(
        "serenityflow.bridge.serenity_api._try_load_ltx_transformer_direct_gpu",
        lambda transformer, **kwargs: direct_gpu.append(kwargs) or True,
    )

    _wrap_official_ltx_ledger_transformer_cpu_stage(ledger, model, stage_label="Official Stage 1")
    transformer = ledger.transformer()

    assert isinstance(transformer, DummyTransformer)
    assert calls == [("factory_device", torch.device("cpu"))]
    assert prepared == [("DummyTransformer", "/tmp/fake.safetensors", "Official Stage 1")]
    assert direct_gpu and direct_gpu[0]["stage_label"] == "Official Stage 1"
    assert ledger.device == torch.device("cuda")


def test_resolve_ltxv_asset_from_hf_cache_uses_snapshot(monkeypatch, tmp_path):
    hub_root = tmp_path / "hf" / "hub"
    snapshot = hub_root / "models--Lightricks--LTX-2.3" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    target = snapshot / "ltx-2.3-22b-distilled-lora-384.safetensors"
    target.write_bytes(b"lora")
    _resolve_ltxv_asset_from_hf_cache.cache_clear()

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub_root))

    assert _resolve_ltxv_asset_from_hf_cache(target.name) == str(target)


def test_resolve_ltxv_asset_falls_back_to_hf_cache(monkeypatch, tmp_path):
    hub_root = tmp_path / "hf" / "hub"
    snapshot = hub_root / "models--Lightricks--LTX-2.3" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    target = snapshot / "ltx-2.3-22b-distilled-lora-384.safetensors"
    target.write_bytes(b"lora")
    _resolve_ltxv_asset_from_hf_cache.cache_clear()

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub_root))
    monkeypatch.setattr(model_paths, "get_model_paths", lambda: SimpleNamespace(find=lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError())))

    assert _resolve_ltxv_asset(None, "loras", target.name) == str(target)


def test_resolve_ltxv_asset_respects_candidate_order_across_model_paths_and_hf_cache(monkeypatch, tmp_path):
    hub_root = tmp_path / "hf" / "hub"
    snapshot = hub_root / "models--Lightricks--LTX-2.3" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    preferred = snapshot / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    preferred.write_bytes(b"preferred")
    fallback = tmp_path / "models" / "upscalers" / "ltx-2-spatial-upscaler-x2-1.0.safetensors"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"fallback")
    _resolve_ltxv_asset_from_hf_cache.cache_clear()

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub_root))

    def fake_find(name, _folder):
        if name == fallback.name:
            return str(fallback)
        raise FileNotFoundError

    monkeypatch.setattr(model_paths, "get_model_paths", lambda: SimpleNamespace(find=fake_find))

    assert _resolve_ltxv_asset(None, "upscalers", preferred.name, fallback.name) == str(preferred)


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


def test_desktop_fast_ltx_checkpoint_candidates_match_23_fp8_names():
    assert _desktop_fast_ltx_checkpoint_candidates("/tmp/ltx-2.3-22b-dev-fp8.safetensors") == (
        "ltx-2.3-22b-distilled.safetensors",
    )
    assert _desktop_fast_ltx_checkpoint_candidates("/tmp/ltx-2.3-22b-distilled-fp8.safetensors") == (
        "ltx-2.3-22b-distilled.safetensors",
    )
    assert _desktop_fast_ltx_checkpoint_candidates("/tmp/ltx-2-19b-dev-fp8.safetensors") == ()


def test_maybe_prefer_desktop_fast_ltx_checkpoint_uses_sibling_checkpoint(tmp_path, monkeypatch):
    requested = tmp_path / "ltx-2.3-22b-dev-fp8.safetensors"
    sibling = tmp_path / "ltx-2.3-22b-distilled.safetensors"
    requested.write_bytes(b"fp8")
    sibling.write_bytes(b"bf16")

    monkeypatch.delenv("SERENITY_LTX_SCALED_FP8_EXPERIMENTAL", raising=False)
    monkeypatch.setattr(
        "serenityflow.bridge.serenity_api._checkpoint_has_weight_scales",
        lambda path: path.endswith("-fp8.safetensors"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=24 * 1024**3),
    )

    assert _maybe_prefer_desktop_fast_ltx_checkpoint(str(requested), "auto") == str(sibling)


def test_maybe_prefer_desktop_fast_ltx_checkpoint_honors_experimental_env(tmp_path, monkeypatch):
    requested = tmp_path / "ltx-2.3-22b-dev-fp8.safetensors"
    requested.write_bytes(b"fp8")

    monkeypatch.setenv("SERENITY_LTX_SCALED_FP8_EXPERIMENTAL", "1")
    monkeypatch.setattr("serenityflow.bridge.serenity_api._checkpoint_has_weight_scales", lambda _path: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=24 * 1024**3),
    )

    assert _maybe_prefer_desktop_fast_ltx_checkpoint(str(requested), "auto") == str(requested)


def test_maybe_prefer_desktop_fast_ltx_checkpoint_uses_exact_non_fp8_fallback(tmp_path, monkeypatch):
    requested = tmp_path / "ltx-2.3-22b-dev-fp8.safetensors"
    requested.write_bytes(b"fp8")
    distilled = tmp_path / "ltx-2.3-22b-distilled.safetensors"
    distilled.write_bytes(b"bf16")

    monkeypatch.delenv("SERENITY_LTX_SCALED_FP8_EXPERIMENTAL", raising=False)
    monkeypatch.setattr("serenityflow.bridge.serenity_api._checkpoint_has_weight_scales", lambda path: path.endswith("-fp8.safetensors"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=24 * 1024**3),
    )
    monkeypatch.setattr(
        "serenityflow.bridge.serenity_api._resolve_ltxv_asset_from_hf_cache",
        lambda name: str(distilled) if name == "ltx-2.3-22b-distilled.safetensors" else None,
    )

    class FakePaths:
        dirs = {"diffusion_models": []}

    monkeypatch.setattr("serenityflow.bridge.model_paths.get_model_paths", lambda: FakePaths())

    assert _maybe_prefer_desktop_fast_ltx_checkpoint(str(requested), "auto") == str(distilled)
