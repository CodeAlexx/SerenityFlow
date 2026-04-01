"""Text encoding, latent operations, sampling, and VAE encode/decode."""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --- Facade proxies (avoid circular import at module level) ---
def _get():
    from serenityflow.bridge.serenity_api import _get as _impl
    return _impl()


from serenityflow.bridge.preview import _make_step_callback
from serenityflow.bridge.loading import CLIPWrapper, VAEWrapper


# ---------------------------------------------------------------------------
# Text Encoding
# ---------------------------------------------------------------------------


def encode_text(clip: CLIPWrapper, text: str) -> list[dict]:
    """Encode text using CLIP. Returns conditioning in ComfyUI format.

    Each entry is a dict with 'cross_attn' and optionally 'pooled_output'.
    """
    # SD3 needs special handling: model expects T5-only context (4096d) and
    # concat(clip_l_pooled, clip_g_pooled) as pooled_projections (2048d).
    # Encode each encoder separately instead of using _encode_sd3 which
    # concatenates all hidden states into 6144d (wrong format for the model).
    _is_sd3 = False
    try:
        from serenity.inference.models.detection import ModelArchitecture
        _is_sd3 = hasattr(clip, "_arch") and clip._arch == ModelArchitecture.SD3
    except ImportError:
        pass

    if _is_sd3:
        try:
            from serenity.inference.text.encoders import TextEncoderType
            clip_l_enc = clip._manager.get_encoder(TextEncoderType.CLIP_L)
            clip_g_enc = clip._manager.get_encoder(TextEncoderType.CLIP_G)
            t5_enc = clip._manager.get_encoder(TextEncoderType.T5_XXL)

            clip_l_out = clip_l_enc.encode(text, clip_skip=clip.clip_skip)
            clip_g_out = clip_g_enc.encode(text, clip_skip=clip.clip_skip)
            t5_out = t5_enc.encode(text)

            cond = t5_out.hidden_states  # (B, seq, 4096)
            pooled = torch.cat([
                clip_l_out.pooled_output,
                clip_g_out.pooled_output,
            ], dim=-1)  # (B, 2048)
            attention_mask = None

            entry = {"cross_attn": cond}
            if pooled is not None:
                entry["pooled_output"] = pooled
            return [entry]
        except Exception as e:
            logger.warning("SD3 direct encode failed, falling back: %s", e)

    try:
        result = clip._manager.encode_for_model(
            clip._arch, text, "",
            clip_skip=clip.clip_skip,
        )
    except Exception as e:
        logger.warning("Text encoding failed: %s. Using zeros.", e, exc_info=True)
        return [{"cross_attn": torch.zeros(1, 77, 768), "pooled_output": torch.zeros(1, 768)}]

    cond = result.get("cond")
    pooled = result.get("pooled")
    if pooled is None:
        pooled = result.get("pooled_output")
    attention_mask = result.get("attention_mask")

    entry = {}
    if cond is not None:
        entry["cross_attn"] = cond
    else:
        entry["cross_attn"] = torch.zeros(1, 77, 768)

    if pooled is not None:
        entry["pooled_output"] = pooled
    if attention_mask is not None:
        entry["attention_mask"] = attention_mask

    # Qwen/Gemma-based image models need the prompt attention mask to recover
    # effective text sequence lengths during sampling.
    try:
        if "attention_mask" in entry:
            raise StopIteration
        from serenity.inference.text.encoders import TextEncoderType

        encoder_type = None
        if getattr(clip, "_arch", None) == clip._arch.__class__.QWEN:
            encoder_type = TextEncoderType.QWEN
        elif getattr(clip, "_arch", None) in (clip._arch.__class__.LUMINA, clip._arch.__class__.ZIMAGE):
            encoder_type = TextEncoderType.GEMMA

        if encoder_type is not None:
            encoder = clip._manager.get_encoder(encoder_type)
            tokenizer = getattr(encoder, "_tokenizer", None)
            if tokenizer is not None:
                tokens = tokenizer(
                    text,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                if "attention_mask" in tokens:
                    entry["attention_mask"] = tokens["attention_mask"]
    except StopIteration:
        pass
    except Exception:
        logger.debug("Prompt attention mask inference skipped", exc_info=True)

    # Extra conditioning (FLUX clip_cond, etc.)
    for key in ("clip_cond", "neg_pooled"):
        if key in result:
            entry[key] = result[key]

    return [entry]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _extract_model_output(raw_out):
    """Extract tensor from model output (may be tuple or ModelOutput)."""
    if isinstance(raw_out, torch.Tensor):
        return raw_out
    if hasattr(raw_out, "sample"):
        return raw_out.sample
    if isinstance(raw_out, tuple):
        return raw_out[0]
    return raw_out


def _is_qwen_transformer(model: nn.Module) -> bool:
    """Best-effort detection for diffusers' Qwen image transformer."""
    return "qwen" in model.__class__.__name__.lower()


def _is_zimage_transformer(model: nn.Module) -> bool:
    """Best-effort detection for diffusers' Z-Image transformer."""
    class_name = model.__class__.__name__.lower()
    if "zimage" in class_name or "z_image" in class_name:
        return True
    arch = getattr(model, "_serenity_arch", None)
    return arch is not None and "zimage" in str(arch).lower()


def _pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack FLUX-style BCHW latents to sequence tokens."""
    batch, channels, height, width = latents.shape
    packed = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(batch, (height // 2) * (width // 2), channels * 4)


def _unpack_flux_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack FLUX-style sequence tokens to BCHW latents."""
    batch, _seq, channels = latents.shape
    latent_channels = channels // 4
    result = latents.view(batch, height // 2, width // 2, latent_channels, 2, 2)
    result = result.permute(0, 3, 1, 4, 2, 5)
    return result.reshape(batch, latent_channels, height, width)


def _patchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Patchify FLUX.2 latents from 32 channels to 128 channels."""
    batch, channels, height, width = latents.shape
    if channels == 128:
        return latents
    if channels != 32:
        raise ValueError(f"FLUX.2 latents must have 32 or 128 channels, got {channels}")
    patched = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    patched = patched.permute(0, 1, 3, 5, 2, 4)
    return patched.reshape(batch, channels * 4, height // 2, width // 2)


def _unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Reverse FLUX.2 patchification back to 32-channel VAE latents."""
    batch, channels, height, width = latents.shape
    if channels == 32:
        return latents
    if channels != 128:
        raise ValueError(f"FLUX.2 patchified latents must have 32 or 128 channels, got {channels}")
    unpacked = latents.view(batch, 32, 2, 2, height, width)
    unpacked = unpacked.permute(0, 1, 4, 2, 5, 3)
    return unpacked.reshape(batch, 32, height * 2, width * 2)


def _pack_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Flatten patchified FLUX.2 spatial latents to a token sequence."""
    batch, channels, height, width = latents.shape
    return latents.permute(0, 2, 3, 1).reshape(batch, height * width, channels)


def _unpack_flux2_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Restore FLUX.2 patchified spatial layout from a token sequence."""
    batch, _seq, channels = latents.shape
    return latents.view(batch, height, width, channels).permute(0, 3, 1, 2)


def _prepare_flux_image_ids(height: int, width: int, device, dtype, t_offset: float = 0.0) -> torch.Tensor:
    """Create FLUX RoPE image ids for a packed latent grid."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 0] = t_offset
    latent_image_ids[..., 1] = torch.arange(height)[:, None]
    latent_image_ids[..., 2] = torch.arange(width)[None, :]
    return latent_image_ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def _prepare_flux2_image_ids(height: int, width: int, device, dtype, t_offset: float = 0.0) -> torch.Tensor:
    """Create FLUX.2 / Klein RoPE ids for a patchified latent grid."""
    h_pos = torch.arange(height, device=device, dtype=dtype)
    w_pos = torch.arange(width, device=device, dtype=dtype)
    h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing="ij")
    t_coord = torch.full_like(h_grid, t_offset)
    l_coord = torch.zeros_like(h_grid)
    return torch.stack([t_coord, h_grid, w_grid, l_coord], dim=-1).reshape(height * width, 4)


def _prepare_flux2_text_ids(text_seq_len: int, device, dtype) -> torch.Tensor:
    """Create FLUX.2 text ids following ComfyUI's txt_ids_dims=[3] convention."""
    txt_ids = torch.zeros((text_seq_len, 4), device=device, dtype=dtype)
    if text_seq_len > 0:
        txt_ids[:, 3] = torch.linspace(
            0,
            text_seq_len - 1,
            steps=text_seq_len,
            device=device,
            dtype=dtype,
        )
    return txt_ids


def _pack_qwen_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack Qwen latents `[B,C,1,H,W]` to sequence tokens."""
    batch, channels, frames, height, width = latents.shape
    if frames != 1:
        raise ValueError(f"Qwen latents must have exactly one frame, got {frames}")
    packed = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(batch, (height // 2) * (width // 2), channels * 4)


def _unpack_qwen_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack Qwen sequence tokens to `[B,C,1,H,W]` latents."""
    batch, _seq, channels = latents.shape
    result = latents.view(batch, height // 2, width // 2, channels // 4, 2, 2)
    result = result.permute(0, 3, 1, 4, 2, 5)
    return result.reshape(batch, channels // 4, 1, height, width)


def _run_sampling(
    model: nn.Module,
    latent: torch.Tensor,
    positive: list[dict],
    negative: list[dict],
    cfg: float,
    sampler_name: str,
    sigmas: torch.Tensor,
    *,
    seed: int = 0,
    add_noise: bool = True,
    noise: torch.Tensor | None = None,
    denoise: float = 1.0,
) -> torch.Tensor:
    """Shared sampler implementation for KSampler and custom sampler nodes."""
    s = _get()
    device = latent.device if latent.is_cuda else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    cond_entry = positive[0] if positive else {}
    uncond_entry = negative[0] if negative else {}

    cond_tensor = cond_entry.get("cross_attn")
    uncond_tensor = uncond_entry.get("cross_attn")
    pooled_cond = cond_entry.get("pooled_output")
    pooled_uncond = uncond_entry.get("pooled_output")
    cond_mask = cond_entry.get("attention_mask")
    uncond_mask = uncond_entry.get("attention_mask")
    reference_latent = cond_entry.get("reference_latent")
    if reference_latent is None:
        reference_latent = uncond_entry.get("reference_latent")

    if cond_tensor is not None:
        cond_tensor = cond_tensor.to(device)
    if uncond_tensor is not None:
        uncond_tensor = uncond_tensor.to(device)
    if pooled_cond is not None:
        pooled_cond = pooled_cond.to(device)
    if pooled_uncond is not None:
        pooled_uncond = pooled_uncond.to(device)
    if cond_mask is not None:
        cond_mask = cond_mask.to(device)
    if uncond_mask is not None:
        uncond_mask = uncond_mask.to(device)

    logger.info(
        "Conditioning shapes: cond=%s, uncond=%s, pooled_cond=%s, pooled_uncond=%s",
        cond_tensor.shape if cond_tensor is not None else None,
        uncond_tensor.shape if uncond_tensor is not None else None,
        pooled_cond.shape if pooled_cond is not None else None,
        pooled_uncond.shape if pooled_uncond is not None else None,
    )

    prediction_type = "eps"
    prediction_kwargs = {}
    if hasattr(model, "_serenity_prediction_type"):
        prediction_type = model._serenity_prediction_type
    elif _is_qwen_transformer(model) or _is_zimage_transformer(model):
        prediction_type = "flow"
    if hasattr(model, "_serenity_model_config"):
        from serenity.inference.models.loader import _get_adapter
        try:
            adapter = _get_adapter(model._serenity_model_config)
        except Exception:
            adapter = None
        if adapter:
            prediction_kwargs = adapter.get_prediction_kwargs()

    is_qwen = _is_qwen_transformer(model)
    is_zimage = _is_zimage_transformer(model)
    _cls_name = model.__class__.__name__
    _is_wan = "Wan" in _cls_name
    # SD3 uses SD3Transformer2DModel; Flux uses FluxTransformer2DModel — both have joint_attention_dim
    is_sd3 = (not is_qwen and not is_zimage and not _is_wan and (
              _cls_name in ("SD3Transformer2DModel",) or
              (prediction_type == "flow" and hasattr(model, "config") and
               hasattr(model.config, "sample_size") and not hasattr(model.config, "in_channels_condition"))))
    is_flux = (not is_qwen and not is_zimage and not is_sd3 and
               hasattr(model, "config") and hasattr(model.config, "joint_attention_dim"))
    is_flux2 = bool(getattr(model, "_serenity_flux2", False))
    is_flux1 = is_flux and not is_flux2

    base_latent = latent.float()
    qwen_input_was_4d = False
    if is_qwen and base_latent.ndim == 4:
        qwen_input_was_4d = True
        base_latent = base_latent.unsqueeze(2)

    if prediction_type == "flow_flux" and base_latent.ndim == 4:
        if is_flux2:
            patched = _patchify_flux2_latents(base_latent)
            actual_seq_len = patched.shape[-2] * patched.shape[-1]
            lat_h, lat_w = patched.shape[-2:]
        else:
            _, _, lat_h, lat_w = base_latent.shape
            actual_seq_len = (lat_h // 2) * (lat_w // 2)
        prediction_kwargs = dict(prediction_kwargs)
        prediction_kwargs["seq_len"] = actual_seq_len
        logger.info("Flux seq_len=%d (from %dx%d latent)", actual_seq_len, lat_h, lat_w)

    prediction = s["get_prediction"](prediction_type, **prediction_kwargs)
    logger.info(
        "Sample: prediction=%s, is_flux=%s, is_qwen=%s, is_zimage=%s, latent=%s, cfg=%.1f",
        prediction_type,
        is_flux,
        is_qwen,
        is_zimage,
        tuple(base_latent.shape),
        cfg,
    )

    sigmas = sigmas.to(device)
    if prediction_type == "flow_flux" and hasattr(prediction, "apply_sigma_shift"):
        sigma_body = prediction.apply_sigma_shift(sigmas[:-1])
        sigmas = torch.cat([sigma_body, sigmas[-1:]])
    elif prediction_type == "flow":
        # Flow matching sigma shift. Read from model attribute (set by ModelSamplingSD3/AuraFlow
        # nodes) or default to 3.0 for SD3.
        shift = getattr(model, "_serenity_sigma_shift", 3.0 if is_sd3 else 0.0)
        if shift > 0:
            body = sigmas[:-1]
            shifted = shift * body / (1.0 + (shift - 1.0) * body)
            sigmas = torch.cat([shifted, sigmas[-1:]])
            logger.info("Flow sigma shift=%.1f applied: [%.4f .. %.4f]",
                         shift, sigmas[0].item(), sigmas[-2].item())

    if noise is None:
        noise = s["create_noise"](
            seed=seed,
            shape=base_latent.shape,
            device="cpu",
            dtype=torch.float32,
        )
    if is_qwen and noise.ndim == 4:
        noise = noise.unsqueeze(2)

    if add_noise:
        if prediction_type in ("flow", "flow_flux"):
            noisy_latent = prediction.noise_scaling(
                sigmas[0].cpu(),
                noise,
                base_latent.cpu(),
                max_denoise=(denoise >= 1.0),
            ).to(device)
        else:
            noisy_latent = (noise * sigmas[0].cpu()).to(device)
    else:
        noisy_latent = base_latent.to(device).float()

    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32

    log_sigmas = None
    _is_flow = prediction_type in ("flow", "flow_flux")
    if not is_flux and not is_qwen and not is_zimage and not is_sd3 and not _is_flow:
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        log_sigmas = ((1 - alphas_cumprod) / alphas_cumprod).sqrt().log().to(device)

    flux_h = flux_w = None
    flux_patch_h = flux_patch_w = None
    flux_img_ids = flux_txt_ids = None
    flux_condition_tokens = None
    if is_flux:
        flux_h = int(base_latent.shape[-2])
        flux_w = int(base_latent.shape[-1])
        txt_len = cond_tensor.shape[1] if cond_tensor is not None else 512
        if is_flux2:
            patched_latent = _patchify_flux2_latents(noisy_latent)
            flux_patch_h = int(patched_latent.shape[-2])
            flux_patch_w = int(patched_latent.shape[-1])
            noisy_latent = _pack_flux2_latents(patched_latent)
            flux_img_ids = _prepare_flux2_image_ids(flux_patch_h, flux_patch_w, device, model_dtype)
            flux_txt_ids = _prepare_flux2_text_ids(txt_len, device, model_dtype)
        else:
            noisy_latent = _pack_flux_latents(noisy_latent)
            flux_patch_h = flux_h // 2
            flux_patch_w = flux_w // 2
            flux_img_ids = _prepare_flux_image_ids(flux_patch_h, flux_patch_w, device, model_dtype)
            flux_txt_ids = torch.zeros(txt_len, 3, device=device, dtype=model_dtype)

        if isinstance(reference_latent, torch.Tensor):
            ref_latent = reference_latent.float()
            if ref_latent.ndim == 5 and ref_latent.shape[2] == 1:
                ref_latent = ref_latent.squeeze(2)
            if ref_latent.ndim == 4 and ref_latent.shape[-2:] == (flux_h, flux_w):
                if is_flux2:
                    ref_latent = _patchify_flux2_latents(ref_latent)
                    ref_patch_h = int(ref_latent.shape[-2])
                    ref_patch_w = int(ref_latent.shape[-1])
                    flux_condition_tokens = _pack_flux2_latents(ref_latent.to(device))
                    ref_img_ids = _prepare_flux2_image_ids(
                        ref_patch_h,
                        ref_patch_w,
                        device,
                        model_dtype,
                        t_offset=10.0,
                    )
                else:
                    flux_condition_tokens = _pack_flux_latents(ref_latent.to(device))
                    ref_img_ids = _prepare_flux_image_ids(
                        flux_patch_h,
                        flux_patch_w,
                        device,
                        model_dtype,
                        t_offset=10.0,
                    )
                flux_img_ids = torch.cat([flux_img_ids, ref_img_ids], dim=0)

    qwen_h = qwen_w = None
    qwen_img_shapes = None
    qwen_condition_tokens = None
    if is_qwen:
        qwen_h = int(base_latent.shape[-2])
        qwen_w = int(base_latent.shape[-1])
        noisy_latent = _pack_qwen_latents(noisy_latent)
        qwen_img_shapes = [[(1, qwen_h // 2, qwen_w // 2)] for _ in range(noisy_latent.shape[0])]
        if cond_entry.get("edit_image") is not None or getattr(model, "_serenity_edit_mode", False):
            qwen_condition_tokens = _pack_qwen_latents(base_latent.to(device))
            qwen_img_shapes = [
                [(1, qwen_h // 2, qwen_w // 2), (1, qwen_h // 2, qwen_w // 2)]
                for _ in range(noisy_latent.shape[0])
            ]

    sdxl_time_ids = None
    if not is_flux and not is_qwen and not is_zimage and base_latent.ndim == 4:
        _, _, lh, lw = base_latent.shape
        target_h, target_w = lh * 8, lw * 8
        sdxl_time_ids = torch.tensor(
            [[target_h, target_w, 0, 0, target_h, target_w]],
            device=device,
            dtype=model_dtype,
        )

    def sigma_to_discrete_timestep(sigma):
        """Convert continuous sigma to discrete diffusers timestep (0-999)."""
        log_sigma = sigma.log()
        dists = log_sigma - log_sigmas
        low_idx = dists.ge(0).cumsum(dim=0).argmax().clamp(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def build_model_kwargs(enc_hidden, pooled, enc_mask):
        """Build model kwargs for this model family."""
        kwargs: dict[str, Any] = {}
        if enc_hidden is not None:
            if is_zimage:
                cap_feats: list[torch.Tensor] = []
                hidden = enc_hidden.to(dtype=model_dtype, device=device)
                mask = enc_mask.to(device) if enc_mask is not None else None
                for batch_idx in range(hidden.shape[0]):
                    tokens = hidden[batch_idx]
                    if mask is not None and batch_idx < mask.shape[0] and mask[batch_idx].shape[0] == tokens.shape[0]:
                        active = mask[batch_idx].bool()
                        if torch.any(active):
                            tokens = tokens[active]
                    cap_feats.append(tokens)
                kwargs["cap_feats"] = cap_feats
            else:
                kwargs["encoder_hidden_states"] = enc_hidden.to(dtype=model_dtype, device=device)
        if is_flux:
            if pooled is not None:
                kwargs["pooled_projections"] = pooled.to(dtype=model_dtype, device=device)
            else:
                kwargs["pooled_projections"] = torch.zeros(1, 768, device=device, dtype=model_dtype)
            flux_guidance = float(cond_entry.get("guidance", 3.5))
            if hasattr(model, "config") and getattr(model.config, "guidance_embeds", True) is False:
                flux_guidance = 0.0
            kwargs["guidance"] = torch.tensor([flux_guidance], device=device, dtype=model_dtype)
            kwargs["img_ids"] = flux_img_ids
            kwargs["txt_ids"] = flux_txt_ids
        elif is_qwen:
            if enc_hidden is not None:
                qwen_hidden = enc_hidden.to(dtype=model_dtype, device=device)
            else:
                qwen_hidden = None
            if enc_mask is not None:
                lengths = enc_mask.sum(dim=1).to(dtype=torch.int64).tolist()
                max_len = max(max(lengths), 1)
                if qwen_hidden is not None:
                    packed_hidden = qwen_hidden.new_zeros((qwen_hidden.shape[0], max_len, qwen_hidden.shape[-1]))
                    packed_mask = enc_mask.new_zeros((enc_mask.shape[0], max_len))
                    for batch_idx, length in enumerate(lengths):
                        active = enc_mask[batch_idx].bool()
                        tokens = qwen_hidden[batch_idx][active]
                        if tokens.shape[0] == 0:
                            tokens = qwen_hidden[batch_idx, :1]
                        token_count = int(tokens.shape[0])
                        packed_hidden[batch_idx, :token_count] = tokens
                        packed_mask[batch_idx, :token_count] = 1
                        lengths[batch_idx] = token_count
                    kwargs["encoder_hidden_states"] = packed_hidden
                    kwargs["encoder_hidden_states_mask"] = packed_mask
                else:
                    kwargs["encoder_hidden_states_mask"] = enc_mask[:, :max_len]
                kwargs["txt_seq_lens"] = lengths
            else:
                if qwen_hidden is not None:
                    kwargs["encoder_hidden_states"] = qwen_hidden
                seq_len = int(enc_hidden.shape[1]) if enc_hidden is not None else 0
                kwargs["txt_seq_lens"] = [seq_len] * noisy_latent.shape[0]
            kwargs["img_shapes"] = qwen_img_shapes
            if bool(getattr(model.config, "guidance_embeds", False)):
                kwargs["guidance"] = torch.full(
                    (noisy_latent.shape[0],),
                    max(cfg, 1.0),
                    device=device,
                    dtype=torch.float32,
                )
        elif is_sd3:
            # SD3 uses pooled_projections directly (not added_cond_kwargs)
            if pooled is not None:
                kwargs["pooled_projections"] = pooled.to(dtype=model_dtype, device=device)
            else:
                kwargs["pooled_projections"] = torch.zeros(1, 2048, device=device, dtype=model_dtype)
        elif pooled is not None:
            pooled_t = pooled.to(dtype=model_dtype, device=device)
            kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled_t,
                "time_ids": sdxl_time_ids,
            }
        return kwargs

    def _invoke_with_retry(fn, *args, **kwargs):
        """Retry model calls after dropping unsupported keyword arguments."""
        attempted: set[str] = set()
        while True:
            try:
                return fn(*args, **kwargs)
            except TypeError as exc:
                msg = str(exc)
                if "unexpected keyword argument" not in msg:
                    raise
                parts = msg.split("'")
                bad_kwarg = parts[1] if len(parts) >= 2 else None
                if bad_kwarg is None or bad_kwarg not in kwargs or bad_kwarg in attempted:
                    raise
                attempted.add(bad_kwarg)
                logger.info(
                    "Dropping unsupported model kwarg '%s' for %s",
                    bad_kwarg,
                    model.__class__.__name__,
                )
                kwargs = dict(kwargs)
                kwargs.pop(bad_kwarg, None)

    def call_model(inp, timestep, **kwargs):
        """Call the underlying model with family-specific timestep handling."""
        if is_sd3:
            # SD3: timestep is already sigma*1000 from prediction.sigma_to_timestep()
            sd3_t = timestep.to(device=device, dtype=model_dtype)
            return _invoke_with_retry(model, inp, timestep=sd3_t, **kwargs)
        if is_flux:
            return _invoke_with_retry(model, inp, timestep=timestep, **kwargs)
        if is_qwen:
            return _invoke_with_retry(model, inp, timestep=timestep / 1000.0, **kwargs)
        if is_zimage:
            if inp.ndim == 4:
                inp = inp.unsqueeze(2)
            latents = [sample.to(dtype=model_dtype, device=device) for sample in inp]
            cap_feats = kwargs.pop("cap_feats", None)
            if cap_feats is None:
                cap_feats = [torch.zeros(1, 2560, device=device, dtype=model_dtype) for _ in latents]
            z_timestep = 1.0 - (timestep.to(device=device, dtype=torch.float32).reshape(-1) / 1000.0)
            if z_timestep.numel() == 1 and len(latents) != 1:
                z_timestep = z_timestep.expand(len(latents))
            return _invoke_with_retry(model, latents, z_timestep, cap_feats, return_dict=True)
        discrete_t = sigma_to_discrete_timestep(timestep) if log_sigmas is not None else timestep
        return _invoke_with_retry(model, inp, discrete_t, **kwargs)

    def prepare_model_output(raw_out, dtype: torch.dtype, channels: int) -> torch.Tensor:
        """Normalize family-specific model outputs to a BCHW/BCHWT tensor."""
        out = _extract_model_output(raw_out)
        if is_zimage:
            out = torch.stack([item.to(device=device, dtype=dtype) for item in out], dim=0)
            if out.ndim == 5 and out.shape[2] == 1:
                out = out.squeeze(2)
            out = -out
        else:
            out = out.to(device=device, dtype=dtype)
        return out[:, :channels, ...]

    use_cfg = (not is_flux1) and uncond_tensor is not None and cfg > 1.0
    logger.info("CFG decision: use_cfg=%s, is_flux1=%s, uncond_tensor=%s, cfg=%.1f, is_sd3=%s",
                use_cfg, is_flux1, uncond_tensor is not None, cfg, is_sd3)

    def denoise_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        model_input = prediction.calculate_input(sigma, x)
        timestep = prediction.sigma_to_timestep(sigma)
        inp = model_input.to(dtype=model_dtype)

        if flux_condition_tokens is not None:
            inp_for_model = torch.cat([inp, flux_condition_tokens.to(dtype=inp.dtype)], dim=1)
        elif qwen_condition_tokens is not None:
            inp_for_model = torch.cat([inp, qwen_condition_tokens.to(dtype=inp.dtype)], dim=1)
        else:
            inp_for_model = inp

        cond_kwargs = build_model_kwargs(cond_tensor, pooled_cond, cond_mask)
        with torch.no_grad():
            raw_out = call_model(inp_for_model, timestep, **cond_kwargs)
            cond_out = prepare_model_output(raw_out, x.dtype, x.shape[1])

        if not use_cfg:
            return prediction.calculate_denoised(sigma, cond_out, x)

        cond_denoised = prediction.calculate_denoised(sigma, cond_out, x)
        uncond_kwargs = build_model_kwargs(uncond_tensor, pooled_uncond, uncond_mask)
        with torch.no_grad():
            raw_out = call_model(inp_for_model, timestep, **uncond_kwargs)
            uncond_out = prepare_model_output(raw_out, x.dtype, x.shape[1])
        uncond_denoised = prediction.calculate_denoised(sigma, uncond_out, x)
        return s["apply_cfg"](cond_denoised, uncond_denoised, cfg)

    # Wrap denoise_fn with Stagehand managed_forward() if the model has a runtime.
    # This gives async prefetch with pinned pool instead of synchronous per-block hooks.
    _stagehand_rt = getattr(model, '_stagehand_runtime', None)
    if _stagehand_rt is not None:
        _base_denoise_fn = denoise_fn
        _step_counter = [0]

        def denoise_fn(x, sigma):
            _stagehand_rt.begin_step(_step_counter[0])
            with _stagehand_rt.managed_forward():
                result = _base_denoise_fn(x, sigma)
            _stagehand_rt.end_step()
            _step_counter[0] += 1
            return result

        logger.info("Sampling with Stagehand managed_forward (async prefetch)")

    step_callback = _make_step_callback(preview_interval=3)

    # Wrap with pipeline counters if available.
    _sampling_counters = None
    try:
        from serenity.inference.sampling.counters import PipelineCounters
        _sampling_counters = PipelineCounters()
        _sampling_counters.start()
        _instrumented_fn = _sampling_counters.wrap_model_fn(denoise_fn)
        _instrumented_cb = _sampling_counters.make_callback(step_callback)
    except ImportError:
        _instrumented_fn = denoise_fn
        _instrumented_cb = step_callback

    result = s["sample"](
        model_fn=_instrumented_fn,
        noise=noisy_latent,
        sigmas=sigmas,
        sampler_type=sampler_name,
        callback=_instrumented_cb,
    )

    if _sampling_counters is not None:
        _sampling_counters.finalize()
        logger.info("Sampling perf:\n%s", _sampling_counters.report())

    # After sampling, evict all Stagehand-managed blocks from GPU so VAE decode
    # has VRAM headroom.
    if _stagehand_rt is not None:
        try:
            from serenityflow.memory.coordinator import get_coordinator
            coord = get_coordinator()
            if coord is not None:
                coord.release_model(id(model))
            model._stagehand_runtime = None
            # Move entire model to CPU and free GPU memory
            for p in model.parameters():
                if p.data.device.type == "cuda":
                    p.data = torch.empty(0, device="cpu")
            torch.cuda.empty_cache()
            free_after = torch.cuda.mem_get_info()[0]
            logger.info("Stagehand: evicted transformer, VRAM free: %.1f GB",
                        free_after / (1024**3))
        except Exception:
            logger.debug("Stagehand cleanup failed", exc_info=True)

    if is_flux and result.ndim == 3 and flux_h is not None and flux_w is not None:
        if is_flux2 and flux_patch_h is not None and flux_patch_w is not None:
            result = _unpack_flux2_latents(result, flux_patch_h, flux_patch_w)
            bn_mean = getattr(model, "_serenity_flux2_bn_mean", None)
            bn_var = getattr(model, "_serenity_flux2_bn_var", None)
            bn_eps = float(getattr(model, "_serenity_flux2_bn_eps", 1e-4))
            if isinstance(bn_mean, torch.Tensor) and isinstance(bn_var, torch.Tensor):
                mean = bn_mean.view(1, -1, 1, 1).to(device=result.device, dtype=result.dtype)
                std = torch.sqrt(
                    bn_var.view(1, -1, 1, 1).to(device=result.device, dtype=result.dtype) + bn_eps
                )
                if mean.shape[1] == result.shape[1]:
                    result = result * std + mean
            result = _unpatchify_flux2_latents(result)
        else:
            result = _unpack_flux_latents(result, flux_h, flux_w)
            result = result + 0.1159
    elif is_zimage and result.ndim == 4:
        result = result + 0.1159
    elif is_qwen and result.ndim == 3 and qwen_h is not None and qwen_w is not None:
        result = _unpack_qwen_latents(result, qwen_h, qwen_w)
        if qwen_input_was_4d:
            result = result.squeeze(2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result.cpu()


def sample(
    model: nn.Module,
    latent: torch.Tensor,
    positive: list[dict],
    negative: list[dict],
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float = 1.0,
    start_step: int | None = None,
    end_step: int | None = None,
    add_noise: bool = True,
    return_with_leftover_noise: bool = False,
) -> torch.Tensor:
    """Run sampling. Returns denoised latent tensor."""
    s = _get()
    prediction_type = getattr(model, "_serenity_prediction_type", None)
    if prediction_type in ("flow", "flow_flux") or _is_qwen_transformer(model):
        sigma_min, sigma_max = 1e-4, 1.0
    else:
        sigma_min, sigma_max = 0.0292, 14.6146

    sigmas = s["compute_sigmas"](
        scheduler=scheduler,
        num_steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    # Handle denoise < 1.0 (img2img): skip early sigmas
    if denoise < 1.0:
        total = len(sigmas) - 1
        skip = int(total * (1.0 - denoise))
        sigmas = sigmas[skip:]

    # Handle start_step / end_step
    if start_step is not None or end_step is not None:
        total = len(sigmas) - 1
        s_start = start_step or 0
        s_end = end_step or total
        sigmas = sigmas[s_start:s_end + 1]

    return _run_sampling(
        model=model,
        latent=latent,
        positive=positive,
        negative=negative,
        cfg=cfg,
        sampler_name=sampler_name,
        sigmas=sigmas,
        seed=seed,
        add_noise=add_noise,
        denoise=denoise,
    )


def sample_custom(
    model: nn.Module,
    latent: torch.Tensor,
    positive: list[dict],
    negative: list[dict],
    cfg: float,
    sampler_name: str,
    sigmas: torch.Tensor,
    *,
    seed: int = 0,
    add_noise: bool = True,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sampling with an explicit sigma schedule and optional custom noise."""
    if not isinstance(sigmas, torch.Tensor):
        sigmas = torch.tensor(sigmas, dtype=torch.float32)
    return _run_sampling(
        model=model,
        latent=latent,
        positive=positive,
        negative=negative,
        cfg=cfg,
        sampler_name=sampler_name,
        sigmas=sigmas,
        seed=seed,
        add_noise=add_noise,
        noise=noise,
    )


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


def vae_decode(vae: VAEWrapper, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent to image. Returns BCHW float32 [0, 1].

    For LTXVModelWrapper, delegates to the video VAE decoder.
    """
    from serenityflow.bridge.ltxv import LTXVModelWrapper
    if isinstance(vae, LTXVModelWrapper):
        return vae.vae_decode(latent)
    if vae is None or vae.decoder is None:
        raise RuntimeError("No VAE decoder available")
    return vae.decoder.decode(latent)


def vae_encode(vae: VAEWrapper, image: torch.Tensor) -> torch.Tensor:
    """Encode image to latent. Expects BCHW float32 [0, 1]."""
    if vae is None or vae.encoder is None:
        raise RuntimeError("No VAE encoder available")
    return vae.encoder.encode(image)


def vae_decode_tiled(vae: VAEWrapper, latent: torch.Tensor, tile_size: int = 512) -> torch.Tensor:
    """Tiled VAE decode for large images."""
    if vae is None or vae.decoder is None:
        raise RuntimeError("No VAE decoder available")
    return vae.decoder.decode(latent, tiling=True, tile_size=tile_size)


def vae_encode_tiled(vae: VAEWrapper, image: torch.Tensor, tile_size: int = 512) -> torch.Tensor:
    """Tiled VAE encode."""
    if vae is None or vae.encoder is None:
        raise RuntimeError("No VAE encoder available")
    return vae.encoder.encode(image, tiling=True, tile_size=tile_size)
