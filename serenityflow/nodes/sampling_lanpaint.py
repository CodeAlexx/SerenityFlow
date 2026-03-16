"""LanPaint inpainting nodes for SerenityFlow.

Three nodes:
- LanPaintKSampler: drop-in KSampler replacement for inpainting
- LanPaintKSamplerAdvanced: full parameter control
- LanPaintInpaint: all-in-one convenience node (image in, image out)
"""
from __future__ import annotations

import logging

import torch

from serenityflow.nodes.registry import registry
from serenityflow.nodes.sampling import _SAMPLER_NAMES, _SCHEDULER_NAMES

log = logging.getLogger(__name__)


def _build_dual_cfg_denoise_fn(
    model, positive, negative, cfg, cfg_big, seed, prediction_type, device,
):
    """Build a denoise function that returns (x_0, x_0_big).

    For LanPaint we need two denoised outputs per model call:
    - x_0: standard CFG result (for unmasked region score)
    - x_0_big: alternate CFG result (for masked region score)

    For Image First mode: cfg_big == cfg (same result)
    For Prompt First mode: cfg_big == -0.5 (heavily prompt-weighted)
    For Flux: no CFG, guidance is an embedding. cfg_big is ignored.
    """
    from serenityflow.bridge.serenity_api import _get, _extract_model_output

    s = _get()

    cond_tensor = positive[0].get("cross_attn") if positive else None
    uncond_tensor = negative[0].get("cross_attn") if negative else None
    pooled_cond = positive[0].get("pooled_output") if positive else None
    pooled_uncond = negative[0].get("pooled_output") if negative else None

    if cond_tensor is not None:
        cond_tensor = cond_tensor.to(device)
    if uncond_tensor is not None:
        uncond_tensor = uncond_tensor.to(device)
    if pooled_cond is not None:
        pooled_cond = pooled_cond.to(device)
    if pooled_uncond is not None:
        pooled_uncond = pooled_uncond.to(device)

    prediction_kwargs = {}
    if hasattr(model, "_serenity_model_config"):
        from serenity.inference.models.loader import _get_adapter
        adapter = _get_adapter(model._serenity_model_config)
        if adapter:
            prediction_kwargs = adapter.get_prediction_kwargs()

    prediction = s["get_prediction"](prediction_type, **prediction_kwargs)
    model_dtype = next(model.parameters()).dtype
    _is_flux = hasattr(model, "config") and hasattr(model.config, "joint_attention_dim")
    _use_cfg = not _is_flux and uncond_tensor is not None and cfg > 1.0

    # Build model kwargs (same as serenity_api.sample)
    _log_sigmas = None
    if not _is_flux:
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        _log_sigmas = ((1 - alphas_cumprod) / alphas_cumprod).sqrt().log().to(device)

    # Build flux position IDs if needed
    _flux_img_ids = None
    _flux_txt_ids = None
    if _is_flux:
        txt_len = cond_tensor.shape[1] if cond_tensor is not None else 512
        _flux_txt_ids = torch.zeros(txt_len, 3, device=device, dtype=model_dtype)

    _sdxl_time_ids = None

    def _build_model_kwargs(enc_hidden, pooled):
        kwargs = {}
        if enc_hidden is not None:
            kwargs["encoder_hidden_states"] = enc_hidden.to(dtype=model_dtype, device=device)
        if _is_flux:
            if pooled is not None:
                kwargs["pooled_projections"] = pooled.to(dtype=model_dtype, device=device)
            else:
                kwargs["pooled_projections"] = torch.zeros(1, 768, device=device, dtype=model_dtype)
            _flux_guidance = 3.5
            if hasattr(model, "config") and getattr(model.config, "guidance_embeds", True) is False:
                _flux_guidance = 0.0
            kwargs["guidance"] = torch.tensor([_flux_guidance], device=device, dtype=model_dtype)
            if _flux_img_ids is not None:
                kwargs["img_ids"] = _flux_img_ids
            kwargs["txt_ids"] = _flux_txt_ids
        elif pooled is not None:
            pooled_t = pooled.to(dtype=model_dtype, device=device)
            kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled_t,
                "time_ids": _sdxl_time_ids,
            }
        return kwargs

    def _sigma_to_timestep(sigma):
        if _log_sigmas is None:
            return sigma
        log_sigma = sigma.log()
        dists = log_sigma - _log_sigmas
        low_idx = dists.ge(0).cumsum(dim=0).argmax().clamp(max=_log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = _log_sigmas[low_idx], _log_sigmas[high_idx]
        w = ((low - log_sigma) / (low - high)).clamp(0, 1)
        return ((1 - w) * low_idx + w * high_idx).view(sigma.shape)

    def _call_model(inp, timestep, **kwargs):
        if _is_flux:
            return model(inp, timestep=timestep, **kwargs)
        else:
            discrete_t = _sigma_to_timestep(timestep) if _log_sigmas is not None else timestep
            return model(inp, discrete_t, **kwargs)

    def _apply_cfg_at_scale(cond_denoised, uncond_denoised, scale):
        """Apply CFG at a specific scale."""
        return uncond_denoised + scale * (cond_denoised - uncond_denoised)

    def dual_denoise_fn(x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_0, x_0_big) — denoised at two CFG scales."""
        model_input = prediction.calculate_input(sigma, x)
        timestep = prediction.sigma_to_timestep(sigma)
        inp = model_input.to(dtype=model_dtype)

        cond_kwargs = _build_model_kwargs(cond_tensor, pooled_cond)

        with torch.no_grad():
            raw_out = _call_model(inp, timestep, **cond_kwargs)
            cond_out = _extract_model_output(raw_out).to(x.dtype)

        if not _use_cfg:
            # Flux or no CFG: both outputs are the same
            denoised = prediction.calculate_denoised(sigma, cond_out, x)
            return denoised, denoised

        cond_denoised = prediction.calculate_denoised(sigma, cond_out, x)

        uncond_kwargs = _build_model_kwargs(uncond_tensor, pooled_uncond)
        with torch.no_grad():
            raw_out = _call_model(inp, timestep, **uncond_kwargs)
            uncond_out = _extract_model_output(raw_out).to(x.dtype)
        uncond_denoised = prediction.calculate_denoised(sigma, uncond_out, x)

        x_0 = _apply_cfg_at_scale(cond_denoised, uncond_denoised, cfg)
        x_0_big = _apply_cfg_at_scale(cond_denoised, uncond_denoised, cfg_big)

        return x_0, x_0_big

    # Store metadata for the solver
    dual_denoise_fn.is_flux = _is_flux
    dual_denoise_fn.is_flow = prediction_type in ("flow", "flow_flux")
    dual_denoise_fn.prediction = prediction
    dual_denoise_fn.prediction_type = prediction_type

    return dual_denoise_fn


def _sample_with_lanpaint(
    model, latent, positive, negative, seed, steps, cfg, sampler_name,
    scheduler, denoise, thinking_steps, lambda_strength, friction,
    step_size, beta, mode, early_stop_threshold, early_stop_patience,
    noise_mask, video=False,
):
    """Core LanPaint sampling implementation."""
    from serenityflow.bridge.serenity_api import _get
    from serenityflow.sampling.lanpaint import LanPaintSolver, prepare_mask, binarize_mask

    s = _get()
    device = latent.device if latent.is_cuda else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Determine prediction type
    prediction_type = "eps"
    if hasattr(model, "_serenity_prediction_type"):
        prediction_type = model._serenity_prediction_type

    is_flux = hasattr(model, "config") and hasattr(model.config, "joint_attention_dim")
    is_flow = prediction_type in ("flow", "flow_flux")

    # Determine cfg_big based on mode
    if mode == "image_first":
        cfg_big = cfg
    else:
        cfg_big = -0.5

    # Build dual-CFG denoise function
    dual_fn = _build_dual_cfg_denoise_fn(
        model, positive, negative, cfg, cfg_big, seed, prediction_type, device,
    )

    # Compute sigmas
    if is_flow:
        sigma_min, sigma_max = 1e-4, 1.0
    else:
        sigma_min, sigma_max = 0.0292, 14.6146

    prediction = dual_fn.prediction
    sigmas = s["compute_sigmas"](
        scheduler=scheduler, num_steps=steps,
        sigma_min=sigma_min, sigma_max=sigma_max,
    ).to(device)

    if prediction_type == "flow_flux" and hasattr(prediction, "apply_sigma_shift"):
        sigma_body = sigmas[:-1]
        sigma_body = prediction.apply_sigma_shift(sigma_body)
        sigmas = torch.cat([sigma_body, sigmas[-1:]])

    if denoise < 1.0:
        total = len(sigmas) - 1
        skip = int(total * (1.0 - denoise))
        sigmas = sigmas[skip:]

    # Create noise
    noise = s["create_noise"](seed=seed, shape=latent.shape, device="cpu", dtype=torch.float32)

    # Prepare mask
    if noise_mask is not None:
        mask = prepare_mask(noise_mask, latent.shape, device, video=video)
        mask = binarize_mask(mask)
    else:
        # No mask — just run standard sampling
        log.warning("LanPaint called without mask, falling back to standard sampling")
        from serenityflow.bridge.serenity_api import sample as std_sample
        return std_sample(
            model=model, latent=latent, positive=positive, negative=negative,
            seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, denoise=denoise,
        )

    # Initialize noisy latent
    latent_on_device = latent.to(device)
    if is_flow:
        noisy_latent = prediction.noise_scaling(
            sigmas[0].cpu(), noise, latent, max_denoise=(denoise >= 1.0),
        ).to(device)
    else:
        noisy_latent = (noise * sigmas[0].cpu()).to(device)

    # Create the LanPaint solver
    solver = LanPaintSolver(
        model_fn=dual_fn,
        n_steps=thinking_steps,
        friction=friction,
        lambda_strength=lambda_strength,
        beta=beta,
        step_size=step_size,
        is_flux=is_flux,
        is_flow=is_flow,
        early_stop_threshold=early_stop_threshold,
        early_stop_patience=early_stop_patience,
    )

    # Build noise scaling function for the replace step
    def noise_scaling_fn(sigma, n, img):
        if is_flow:
            s_reshaped = sigma.reshape([sigma.shape[0]] + [1] * (len(n.shape) - 1))
            return s_reshaped * n + (1.0 - s_reshaped) * img
        else:
            s_reshaped = sigma.reshape([sigma.shape[0]] + [1] * (len(n.shape) - 1))
            return img + n * s_reshaped

    # Noise for the known-region replace step
    solver_noise = noise.to(device)

    # latent_mask convention: 1 = known (will be replaced from original),
    # 0 = unknown (will be generated). This is the INVERSE of user's "paint mask".
    # In ComfyUI, noise_mask=1 means "denoise this pixel" (unknown),
    # so latent_mask = 1 - noise_mask.
    latent_mask = 1 - mask

    # Run the denoising loop with LanPaint at each step
    x = noisy_latent
    total_steps = len(sigmas) - 1

    for i in range(total_steps):
        sigma = sigmas[i:i+1]
        sigma_next = sigmas[i+1:i+2]

        # Determine thinking steps: more on early steps, fewer on later steps
        current_thinking = thinking_steps

        # Run LanPaint solver for this step
        denoised = solver(
            x=x,
            sigma=sigma,
            latent_image=latent_on_device,
            noise=solver_noise,
            latent_mask=latent_mask,
            noise_scaling_fn=noise_scaling_fn,
            n_steps=current_thinking,
        )

        # Step to next noise level (simple Euler step)
        if sigma_next > 0:
            if is_flow:
                # Flow matching step
                d = (x - denoised) / sigma
                x = x + d * (sigma_next - sigma)
            else:
                # Standard diffusion step
                d = (x - denoised) / sigma
                x = denoised + d * sigma_next
        else:
            x = denoised

        log.debug("LanPaint step %d/%d, sigma=%.4f->%.4f", i + 1, total_steps,
                  sigma.item(), sigma_next.item() if sigma_next.numel() > 0 else 0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return x.cpu()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@registry.register(
    "LanPaintKSampler",
    return_types=("LATENT",),
    category="sampling",
    display_name="LanPaint KSampler",
    input_types={
        "required": {
            "model": ("MODEL",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**53}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (_SAMPLER_NAMES,),
            "scheduler": (_SCHEDULER_NAMES,),
            "positive": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "thinking_steps": ("INT", {"default": 5, "min": 2, "max": 50,
                "tooltip": "Langevin reasoning iterations per denoise step. More = better quality, slower."}),
            "mode": (["image_first", "prompt_first"],
                {"tooltip": "image_first: emphasize image quality. prompt_first: emphasize prompt adherence."}),
        },
        "optional": {
            "negative": ("CONDITIONING",),
        },
    },
)
def lanpaint_ksampler(
    model, seed, steps, cfg, sampler_name, scheduler,
    positive, latent_image, denoise=1.0, thinking_steps=5,
    mode="image_first", negative=None,
):
    from serenityflow.bridge.types import unwrap_latent, wrap_latent

    latent = unwrap_latent(latent_image)
    noise_mask = latent_image.get("noise_mask") if isinstance(latent_image, dict) else None

    result = _sample_with_lanpaint(
        model=model, latent=latent, positive=positive, negative=negative,
        seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
        scheduler=scheduler, denoise=denoise,
        thinking_steps=thinking_steps, lambda_strength=7.0, friction=15.0,
        step_size=0.3, beta=1.0, mode=mode,
        early_stop_threshold=0.0, early_stop_patience=1,
        noise_mask=noise_mask,
    )

    return (wrap_latent(result),)


@registry.register(
    "LanPaintKSamplerAdvanced",
    return_types=("LATENT",),
    category="sampling",
    display_name="LanPaint KSampler (Advanced)",
    input_types={
        "required": {
            "model": ("MODEL",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**53}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (_SAMPLER_NAMES,),
            "scheduler": (_SCHEDULER_NAMES,),
            "positive": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "thinking_steps": ("INT", {"default": 5, "min": 2, "max": 50,
                "tooltip": "Langevin reasoning iterations per denoise step."}),
            "lambda_strength": ("FLOAT", {"default": 7.0, "min": 4.0, "max": 10.0, "step": 0.1,
                "tooltip": "Content alignment strength. Higher = more aligned with known region."}),
            "friction": ("FLOAT", {"default": 15.0, "min": 10.0, "max": 20.0, "step": 0.1,
                "tooltip": "Langevin dynamics friction. Lower = faster convergence but less stable."}),
            "step_size": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.5, "step": 0.01,
                "tooltip": "Langevin step size. Larger = faster but potentially unstable."}),
            "beta": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                "tooltip": "Step size ratio between masked/unmasked regions."}),
            "mode": (["image_first", "prompt_first"],),
            "early_stop_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001,
                "tooltip": "Convergence threshold for early stopping. 0 = disabled."}),
            "early_stop_patience": ("INT", {"default": 1, "min": 1, "max": 100,
                "tooltip": "Consecutive stable steps before stopping."}),
            "semantic_early_stop": ("BOOLEAN", {"default": False,
                "tooltip": "Use model-based convergence check instead of simple distance metric."}),
        },
        "optional": {
            "negative": ("CONDITIONING",),
        },
    },
)
def lanpaint_ksampler_advanced(
    model, seed, steps, cfg, sampler_name, scheduler,
    positive, latent_image, denoise=1.0,
    thinking_steps=5, lambda_strength=7.0, friction=15.0,
    step_size=0.3, beta=1.0, mode="image_first",
    early_stop_threshold=0.0, early_stop_patience=1,
    semantic_early_stop=False,
    negative=None,
):
    from serenityflow.bridge.types import unwrap_latent, wrap_latent

    latent = unwrap_latent(latent_image)
    noise_mask = latent_image.get("noise_mask") if isinstance(latent_image, dict) else None

    result = _sample_with_lanpaint(
        model=model, latent=latent, positive=positive, negative=negative,
        seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
        scheduler=scheduler, denoise=denoise,
        thinking_steps=thinking_steps, lambda_strength=lambda_strength,
        friction=friction, step_size=step_size, beta=beta, mode=mode,
        early_stop_threshold=early_stop_threshold,
        early_stop_patience=early_stop_patience,
        noise_mask=noise_mask,
    )

    return (wrap_latent(result),)


@registry.register(
    "LanPaintInpaint",
    return_types=("IMAGE",),
    category="sampling",
    display_name="LanPaint Inpaint",
    input_types={
        "required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "positive": ("STRING", {"default": "", "multiline": True}),
            "negative": ("STRING", {"default": "", "multiline": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**53}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (_SAMPLER_NAMES,),
            "scheduler": (_SCHEDULER_NAMES,),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "thinking_steps": ("INT", {"default": 5, "min": 2, "max": 50}),
            "mode": (["image_first", "prompt_first"],),
        },
    },
)
def lanpaint_inpaint(
    image, mask, model, clip, vae, positive, negative,
    seed, steps, cfg, sampler_name, scheduler, denoise=1.0,
    thinking_steps=5, mode="image_first",
):
    """All-in-one LanPaint inpainting: image in, inpainted image out."""
    from serenityflow.bridge.serenity_api import encode_text, vae_encode, vae_decode
    from serenityflow.bridge.types import bhwc_to_bchw, bchw_to_bhwc, wrap_latent

    # Encode text
    pos_cond = encode_text(clip, positive)
    neg_cond = encode_text(clip, negative) if negative else None

    # VAE encode image
    image_bchw = bhwc_to_bchw(image)
    latent = vae_encode(vae, image_bchw)

    # Run LanPaint
    result = _sample_with_lanpaint(
        model=model, latent=latent, positive=pos_cond, negative=neg_cond,
        seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
        scheduler=scheduler, denoise=denoise,
        thinking_steps=thinking_steps, lambda_strength=7.0, friction=15.0,
        step_size=0.3, beta=1.0, mode=mode,
        early_stop_threshold=0.0, early_stop_patience=1,
        noise_mask=mask,
    )

    # VAE decode
    decoded = vae_decode(vae, result)
    return (bchw_to_bhwc(decoded),)
