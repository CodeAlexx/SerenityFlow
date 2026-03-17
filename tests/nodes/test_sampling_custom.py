"""Regression tests for custom sampling and workflow-facing loader variants."""
from __future__ import annotations

from unittest.mock import patch

import torch

from serenityflow.nodes.registry import registry

import serenityflow.nodes.loaders  # noqa: F401
import serenityflow.nodes.model_specific.flux  # noqa: F401
import serenityflow.nodes.sampling_custom  # noqa: F401


class TestSamplerCustom:
    def test_sampler_custom_calls_bridge(self, monkeypatch):
        captured = {}

        def fake_sample_custom(**kwargs):
            captured.update(kwargs)
            return kwargs["latent"] + 1

        monkeypatch.setattr("serenityflow.bridge.serenity_api.sample_custom", fake_sample_custom)

        fn = registry.get_function("SamplerCustom")
        latent = {"samples": torch.zeros(1, 4, 8, 8)}
        positive = [{"cross_attn": torch.randn(1, 4, 8)}]
        negative = []
        sigmas = torch.tensor([1.0, 0.0])

        output, denoised = fn(
            model="model-handle",
            add_noise=True,
            noise_seed=123,
            cfg=3.0,
            positive=positive,
            negative=negative,
            sampler={"type": "heun"},
            sigmas=sigmas,
            latent_image=latent,
        )

        assert captured["model"] == "model-handle"
        assert captured["sampler_name"] == "heun"
        assert captured["seed"] == 123
        assert captured["add_noise"] is True
        assert torch.equal(output["samples"], torch.ones_like(latent["samples"]))
        assert torch.equal(denoised["samples"], output["samples"])

    def test_sampler_custom_advanced_accepts_reference_latent(self, monkeypatch):
        captured = {}

        def fake_sample_custom(**kwargs):
            captured.update(kwargs)
            return kwargs["latent"] + 2

        monkeypatch.setattr("serenityflow.bridge.serenity_api.sample_custom", fake_sample_custom)

        fn = registry.get_function("SamplerCustomAdvanced")
        reference_latent = torch.zeros(1, 4, 8, 8)
        output, denoised = fn(
            noise={"type": "empty"},
            guider={"type": "basic", "model": "model-handle", "positive": [{"reference_latent": reference_latent}]},
            sampler={"type": "euler"},
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=[{"reference_latent": reference_latent}],
        )

        assert captured["model"] == "model-handle"
        assert captured["cfg"] == 1.0
        assert captured["negative"] == []
        assert captured["add_noise"] is False
        assert torch.equal(captured["latent"], reference_latent)
        assert torch.equal(captured["noise"], torch.zeros_like(reference_latent))
        assert torch.equal(output["samples"], reference_latent + 2)
        assert torch.equal(denoised["samples"], output["samples"])


class TestWorkflowFacingLoaders:
    @patch("serenityflow.bridge.model_paths.get_model_paths")
    @patch("serenityflow.bridge.serenity_api.load_clip")
    def test_clip_loader_accepts_qwen_device_hint(self, mock_load_clip, mock_paths):
        from serenityflow.nodes.loaders import clip_loader

        mock_paths.return_value.find = lambda name, folder: f"/fake/{name}"
        mock_load_clip.return_value = "clip-handle"

        (clip,) = clip_loader("qwen.safetensors", type="qwen", device="cpu")

        assert clip == "clip-handle"
        mock_load_clip.assert_called_once_with("/fake/qwen.safetensors", clip_type="qwen")

    @patch("serenityflow.bridge.model_paths.get_model_paths")
    @patch("serenityflow.bridge.serenity_api.load_clip")
    def test_clip_loader_passthroughs_flux2_repo_id_when_lookup_misses(self, mock_load_clip, mock_paths):
        from serenityflow.nodes.loaders import clip_loader

        mock_paths.return_value.find.side_effect = FileNotFoundError("missing")
        mock_load_clip.return_value = "clip-handle"

        (clip,) = clip_loader("black-forest-labs/FLUX.2-dev", type="flux2", device="cpu")

        assert clip == "clip-handle"
        mock_load_clip.assert_called_once_with("black-forest-labs/FLUX.2-dev", clip_type="flux2")

    @patch("serenityflow.bridge.model_paths.get_model_paths")
    @patch("serenityflow.bridge.serenity_api.load_dual_clip")
    def test_dual_clip_loader_accepts_device_hint(self, mock_load_dual_clip, mock_paths):
        from serenityflow.nodes.loaders import dual_clip_loader

        mock_paths.return_value.find.side_effect = lambda name, folder: f"/fake/{name}"
        mock_load_dual_clip.return_value = "dual-clip-handle"

        (clip,) = dual_clip_loader(
            "clip_l.safetensors",
            "t5xxl.safetensors",
            type="flux",
            device="cuda",
        )

        assert clip == "dual-clip-handle"
        mock_load_dual_clip.assert_called_once_with(
            "/fake/clip_l.safetensors",
            "/fake/t5xxl.safetensors",
            clip_type="flux",
        )

    @patch("serenityflow.bridge.model_paths.get_model_paths")
    @patch("serenityflow.bridge.serenity_api.load_diffusion_model")
    def test_unet_loader_passthroughs_explicit_path_when_lookup_misses(self, mock_load_model, mock_paths, tmp_path):
        from serenityflow.nodes.loaders import unet_loader

        model_path = tmp_path / "zimage.safetensors"
        model_path.touch()
        mock_paths.return_value.find.side_effect = FileNotFoundError("missing")
        mock_load_model.return_value = "model-handle"

        (model,) = unet_loader(str(model_path), weight_dtype="default")

        assert model == "model-handle"
        mock_load_model.assert_called_once_with(str(model_path), dtype="default")

    def test_clip_loader_registry_exposes_flux2_and_klein_types(self):
        node = registry.get("CLIPLoader")
        clip_types = node.input_types["required"]["type"][0]

        assert "flux2" in clip_types
        assert "klein" in clip_types


class TestFlux2Scheduler:
    def test_accepts_optional_model_input(self):
        node = registry.get("Flux2Scheduler")
        assert "model" in node.input_types.get("optional", {})

        fn = registry.get_function("Flux2Scheduler")
        (sigmas,) = fn(steps=4, model=object(), shift=1.2, denoise=0.5)

        assert sigmas.shape[0] == 5
        assert sigmas[0] > sigmas[-1]


class TestFlux2Latent:
    def test_empty_flux2_latent_uses_32_channel_vae_space(self):
        fn = registry.get_function("EmptyFlux2LatentImage")

        (latent,) = fn(width=1024, height=1024, batch_size=1)

        assert latent["samples"].shape == (1, 32, 128, 128)
