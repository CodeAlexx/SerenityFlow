"""Prompt enhancement for video generation.

Enriches short user prompts into detailed cinematic descriptions using a
small LLM (e.g. Qwen2.5-3B-Instruct) before they are encoded by the
text encoder.  Works standalone with no imports from other serenity modules.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "enhance_prompt",
    "enhance_prompt_standalone",
    "load_enhancer_model",
    "T2V_SYSTEM_PROMPT",
    "I2V_SYSTEM_PROMPT",
]

# ---------------------------------------------------------------------------
# System prompts (adapted from Wan2GP / LTX-2 reference implementations)
# ---------------------------------------------------------------------------

T2V_SYSTEM_PROMPT = (
    "You are an expert cinematic director with many award winning movies. "
    "When writing prompts based on the user input, focus on detailed, "
    "chronological descriptions of actions and scenes.\n"
    "Include specific movements, appearances, camera angles, and "
    "environmental details - all in a single flowing paragraph.\n"
    "Start directly with the action, and keep descriptions literal and precise.\n"
    "Think like a cinematographer describing a shot list.\n"
    "Do not change the user input intent, just enhance it.\n"
    "Keep within 150 words.\n"
    "For best results, build your prompts using this structure:\n"
    "Start with main action in a single sentence\n"
    "Add specific details about movements and gestures\n"
    "Describe character/object appearances precisely\n"
    "Include background and environment details\n"
    "Specify camera angles and movements\n"
    "Describe lighting and colors\n"
    "Note any changes or sudden events\n"
    "Do not exceed the 150 word limit!\n"
    "Output the enhanced prompt only."
)

I2V_SYSTEM_PROMPT = (
    "You are an expert cinematic director with many award winning movies.\n"
    "You have been provided with a caption of an image of a subject that "
    "relates to the scene to film.\n"
    "Focus on detailed, chronological descriptions of actions and scenes.\n"
    "Include specific movements, appearances, camera angles, and "
    "environmental details - all in a single flowing paragraph.\n"
    "Start directly with the action, and keep descriptions literal and precise.\n"
    "Think like a cinematographer describing a shot list.\n"
    "Keep within 150 words.\n"
    "For best results, build your prompts using this structure:\n"
    "Describe the initial scene first using the image caption of the subject "
    "and then describe how the scene should naturally evolve.\n"
    "Start with main action in a single sentence\n"
    "Add specific details about movements and gestures\n"
    "Describe character/object appearances precisely\n"
    "Include background and environment details\n"
    "Specify camera angles and movements\n"
    "Describe lighting and colors\n"
    "Note any changes or sudden events\n"
    "Do not exceed the 150 word limit!\n"
    "Output the enhanced prompt only."
)

# Minimum prompt length (characters) to consider "already detailed enough".
_LONG_PROMPT_THRESHOLD = 300


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_enhancer_model(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    """Load a small LLM for prompt enhancement.

    Args:
        model_path: Local path (or HF hub id) to a causal LM such as
            ``Qwen/Qwen2.5-3B-Instruct``.
        device: Target device.  When *load_in_4bit* is ``True`` the model
            is loaded directly onto ``cuda`` regardless of this argument.
        dtype: Floating-point dtype for non-quantised weights.
        load_in_4bit: Use BitsAndBytes 4-bit quantisation to reduce VRAM.

    Returns:
        ``(model, tokenizer)`` tuple ready for :func:`enhance_prompt`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # BnB handles device placement internally.
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if not load_in_4bit:
        model = model.to(device)

    model.eval()

    logger.info(
        "Loaded prompt enhancer from %s (4-bit=%s, device=%s)",
        model_path,
        load_in_4bit,
        next(model.parameters()).device,
    )
    return model, tokenizer


def enhance_prompt(
    prompt: str,
    mode: str = "t2v",
    model: Any = None,
    tokenizer: Any = None,
    max_new_tokens: int = 200,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> str:
    """Enhance a short user prompt into a detailed cinematic description.

    If the prompt already exceeds :data:`_LONG_PROMPT_THRESHOLD` characters
    it is returned unchanged (assumed to be sufficiently detailed).

    Args:
        prompt: Raw user prompt.
        mode: ``"t2v"`` (text-to-video) or ``"i2v"`` (image-to-video).
        model: Pre-loaded causal LM (e.g. from :func:`load_enhancer_model`).
            When ``None`` the prompt is returned as-is.
        tokenizer: Matching tokenizer for *model*.
        max_new_tokens: Maximum tokens to generate.
        device: Device hint (used only when *model* is ``None``).
        dtype: Dtype hint (unused when a model is already loaded).

    Returns:
        The enhanced prompt string, or the original prompt when enhancement
        is skipped.
    """
    if len(prompt) > _LONG_PROMPT_THRESHOLD:
        logger.info(
            "Prompt already long (%d chars > %d), skipping enhancement.",
            len(prompt),
            _LONG_PROMPT_THRESHOLD,
        )
        return prompt

    if model is None or tokenizer is None:
        logger.warning(
            "No enhancer model/tokenizer provided; returning original prompt."
        )
        return prompt

    system_prompt = I2V_SYSTEM_PROMPT if mode == "i2v" else T2V_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"user_prompt: {prompt}"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Strip the input tokens to get only the generated portion.
    generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
    enhanced = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not enhanced:
        logger.warning("Enhancement produced empty output; returning original prompt.")
        return prompt

    logger.info("Enhanced prompt (%d -> %d chars): %s", len(prompt), len(enhanced), enhanced)
    return enhanced


def enhance_prompt_standalone(
    prompt: str,
    model_path: str,
    mode: str = "t2v",
    device: str = "cuda",
) -> str:
    """Convenience wrapper: load model, enhance prompt, then unload.

    Useful when no model is pre-loaded and you only need a single
    enhancement.  For repeated calls prefer :func:`load_enhancer_model`
    once and pass the model to :func:`enhance_prompt`.

    Args:
        prompt: Raw user prompt.
        model_path: Path to the enhancer LLM weights.
        mode: ``"t2v"`` or ``"i2v"``.
        device: Target device (``"cuda"`` recommended).

    Returns:
        The enhanced prompt string.
    """
    if len(prompt) > _LONG_PROMPT_THRESHOLD:
        logger.info(
            "Prompt already long (%d chars > %d), skipping enhancement.",
            len(prompt),
            _LONG_PROMPT_THRESHOLD,
        )
        return prompt

    model, tokenizer = load_enhancer_model(
        model_path,
        device=device,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    try:
        return enhance_prompt(
            prompt,
            mode=mode,
            model=model,
            tokenizer=tokenizer,
        )
    finally:
        # Explicit cleanup to free VRAM immediately.
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded prompt enhancer model.")
