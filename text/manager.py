"""Text encoder manager -- model-specific encoding orchestration.

Routes each model architecture to the correct text encoder(s) and
provides a unified API for prompt encoding.

CORRECTED MAPPING (vs the broken serenity/inference/text/encoders.py):
- Klein 4B/9B: Qwen3 mode=klein (stacked layers [9,18,27])
- ZImage: Qwen3 mode=zimage (penultimate layer, dim=2560)
- Flux 2 Dev: Mistral (NOT CLIP+T5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from models.detection import ModelArchitecture
from text.clip import CLIPEncoder, TextOutput
from text.mistral import MistralEncoder
from text.qwen25vl import Qwen25VLEncoder
from text.qwen3 import Qwen3Encoder
from text.t5 import T5Encoder
from text.umt5 import UMT5Encoder

__all__ = [
    "TextEncoderManager",
    "TextEncoderType",
    "get_default_encoder_path",
    "get_required_encoders",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder type enum
# ---------------------------------------------------------------------------


class TextEncoderType(str, Enum):
    """Supported text encoder variants."""

    CLIP_L = "clip_l"
    CLIP_G = "clip_g"
    T5_XXL = "t5_xxl"
    QWEN3_KLEIN = "qwen3_klein"
    QWEN3_ZIMAGE = "qwen3_zimage"
    QWEN25_VL = "qwen25_vl"
    MISTRAL = "mistral"
    UMT5_XXL = "umt5_xxl"


# ---------------------------------------------------------------------------
# Model-to-encoder mapping (CORRECTED)
# ---------------------------------------------------------------------------

_MODEL_ENCODERS: dict[ModelArchitecture, list[TextEncoderType]] = {
    # SD 1.5: single CLIP-L
    ModelArchitecture.SD15: [TextEncoderType.CLIP_L],
    # SD 2.x fallback path in Serenity (approximate: CLIP-L style interface)
    ModelArchitecture.SD20: [TextEncoderType.CLIP_L],
    ModelArchitecture.SD21: [TextEncoderType.CLIP_L],
    # SDXL: dual CLIP (L + G)
    ModelArchitecture.SDXL: [TextEncoderType.CLIP_L, TextEncoderType.CLIP_G],
    ModelArchitecture.SDXL_REFINER: [TextEncoderType.CLIP_G],
    # SD3: triple (CLIP-L + CLIP-G + T5-XXL)
    ModelArchitecture.SD3: [TextEncoderType.CLIP_L, TextEncoderType.CLIP_G, TextEncoderType.T5_XXL],
    # Flux 1: CLIP-L (pooled) + T5-XXL (main)
    ModelArchitecture.FLUX_DEV: [TextEncoderType.CLIP_L, TextEncoderType.T5_XXL],
    ModelArchitecture.FLUX_FILL: [TextEncoderType.CLIP_L, TextEncoderType.T5_XXL],
    ModelArchitecture.FLUX_SCHNELL: [TextEncoderType.CLIP_L, TextEncoderType.T5_XXL],
    # Flux 2 Dev: Mistral3 (stacked layers [10,20,30])
    ModelArchitecture.FLUX_2_DEV: [TextEncoderType.MISTRAL],
    # Flux 2 Klein: Qwen3 mode=klein (stacked layers [9,18,27])
    ModelArchitecture.FLUX_2_KLEIN_4B: [TextEncoderType.QWEN3_KLEIN],
    ModelArchitecture.FLUX_2_KLEIN_9B: [TextEncoderType.QWEN3_KLEIN],
    # Chroma: T5-XXL only (distilled Flux variant)
    ModelArchitecture.CHROMA: [TextEncoderType.T5_XXL],
    # Wan: UMT5-XXL (different vocab/tokenizer than standard T5-XXL)
    ModelArchitecture.WAN: [TextEncoderType.UMT5_XXL],
    # ZImage: Qwen3 mode=zimage (penultimate layer, dim=2560)
    ModelArchitecture.ZIMAGE: [TextEncoderType.QWEN3_ZIMAGE],
    # Qwen Image: Qwen2.5-VL
    ModelArchitecture.QWEN: [TextEncoderType.QWEN25_VL],
    # Lumina uses the same text path as ZImage in Serenity's NextDiT integration.
    ModelArchitecture.LUMINA: [TextEncoderType.QWEN3_ZIMAGE],
    # OneTrainer parity families (best-effort encoder routing).
    ModelArchitecture.PIXART_ALPHA: [TextEncoderType.T5_XXL],
    ModelArchitecture.PIXART_SIGMA: [TextEncoderType.T5_XXL],
    ModelArchitecture.SANA: [TextEncoderType.T5_XXL],
    ModelArchitecture.HUNYUAN_VIDEO: [TextEncoderType.T5_XXL],
    ModelArchitecture.WUERSTCHEN: [TextEncoderType.CLIP_L],
    ModelArchitecture.STABLE_CASCADE: [TextEncoderType.CLIP_L],
    ModelArchitecture.HIDREAM: [TextEncoderType.T5_XXL],
    ModelArchitecture.OMNIGEN: [TextEncoderType.CLIP_L],
    ModelArchitecture.KANDINSKY: [TextEncoderType.CLIP_L],
}


def get_required_encoders(
    model_architecture: ModelArchitecture,
) -> list[TextEncoderType]:
    """Return the list of text encoder types required by *model_architecture*."""
    return list(_MODEL_ENCODERS.get(model_architecture, []))


# ---------------------------------------------------------------------------
# Default HuggingFace encoder paths
# ---------------------------------------------------------------------------


@dataclass
class _EncoderPath:
    """Descriptor for locating a text encoder on HuggingFace."""

    repo: str
    subfolder: str | None = None
    tokenizer_subfolder: str | None = None
    use_projection: bool = False


# Canonical repos shared across architectures.
_CLIP_L_REPO = "openai/clip-vit-large-patch14"
_SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
_T5_XXL_REPO = "google/t5-v1_1-xxl"

_DEFAULT_ENCODER_PATHS: dict[tuple[ModelArchitecture, TextEncoderType], _EncoderPath] = {
    # SD 1.5
    (ModelArchitecture.SD15, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    # SDXL
    (ModelArchitecture.SDXL, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    (ModelArchitecture.SDXL, TextEncoderType.CLIP_G): _EncoderPath(
        repo=_SDXL_REPO,
        subfolder="text_encoder_2",
        tokenizer_subfolder="tokenizer_2",
        use_projection=True,
    ),
    # SDXL Refiner
    (ModelArchitecture.SDXL_REFINER, TextEncoderType.CLIP_G): _EncoderPath(
        repo=_SDXL_REPO,
        subfolder="text_encoder_2",
        tokenizer_subfolder="tokenizer_2",
        use_projection=True,
    ),
    # SD3
    (ModelArchitecture.SD3, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    (ModelArchitecture.SD3, TextEncoderType.CLIP_G): _EncoderPath(
        repo=_SDXL_REPO,
        subfolder="text_encoder_2",
        tokenizer_subfolder="tokenizer_2",
        use_projection=True,
    ),
    (ModelArchitecture.SD3, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    # Flux 1 Dev
    (ModelArchitecture.FLUX_DEV, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    (ModelArchitecture.FLUX_DEV, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    # Flux Fill Dev
    (ModelArchitecture.FLUX_FILL, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    (ModelArchitecture.FLUX_FILL, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    # Flux 1 Schnell
    (ModelArchitecture.FLUX_SCHNELL, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    (ModelArchitecture.FLUX_SCHNELL, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    # Chroma
    (ModelArchitecture.CHROMA, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    # Wan (UMT5-XXL, different from standard T5-XXL)
    (ModelArchitecture.WAN, TextEncoderType.UMT5_XXL): _EncoderPath(repo="google/umt5-xxl"),
    # Qwen Image
    (ModelArchitecture.QWEN, TextEncoderType.QWEN25_VL): _EncoderPath(
        repo="Qwen/Qwen2.5-VL-7B-Instruct",
    ),
    # Lumina (shares Qwen3 text path in this implementation)
    (ModelArchitecture.LUMINA, TextEncoderType.QWEN3_ZIMAGE): _EncoderPath(
        repo="Qwen/Qwen3-4B",
    ),
    # SD2.x (fallback path)
    (ModelArchitecture.SD20, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    (ModelArchitecture.SD21, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    # PixArt/Sana/Hunyuan/HiDream (best-effort default to T5-XXL)
    (ModelArchitecture.PIXART_ALPHA, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    (ModelArchitecture.PIXART_SIGMA, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    (ModelArchitecture.SANA, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    (ModelArchitecture.HUNYUAN_VIDEO, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    (ModelArchitecture.HIDREAM, TextEncoderType.T5_XXL): _EncoderPath(repo=_T5_XXL_REPO),
    # Wuerstchen / Cascade / OmniGen / Kandinsky (best-effort CLIP-L path)
    (ModelArchitecture.WUERSTCHEN, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    (ModelArchitecture.STABLE_CASCADE, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    (ModelArchitecture.OMNIGEN, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    (ModelArchitecture.KANDINSKY, TextEncoderType.CLIP_L): _EncoderPath(repo=_CLIP_L_REPO),
    # NOTE: Klein and ZImage encoder paths depend on the specific model checkpoint
    # directory structure (text encoder is bundled). These are left without defaults
    # because the user must specify the path from the model directory.
}


def get_default_encoder_path(
    architecture: ModelArchitecture,
    encoder_type: TextEncoderType,
) -> _EncoderPath | None:
    """Return the default HuggingFace path for an encoder, or ``None``."""
    return _DEFAULT_ENCODER_PATHS.get((architecture, encoder_type))


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


# Type alias for all supported encoder classes
_AnyEncoder = CLIPEncoder | T5Encoder | UMT5Encoder | Qwen3Encoder | Qwen25VLEncoder | MistralEncoder


class TextEncoderManager:
    """Manage text encoders for a given model architecture.

    Holds cached encoder instances and provides a high-level API to
    load the right encoders for a model, encode prompts according to
    model-specific conventions, and release resources.
    """

    def __init__(self) -> None:
        self._encoders: dict[TextEncoderType, _AnyEncoder] = {}

    # -- encoder access ------------------------------------------------------

    def get_encoder(
        self,
        encoder_type: TextEncoderType,
    ) -> _AnyEncoder:
        """Return a cached encoder, creating an unloaded stub if absent."""
        if encoder_type not in self._encoders:
            if encoder_type == TextEncoderType.CLIP_L:
                self._encoders[encoder_type] = CLIPEncoder()
            elif encoder_type == TextEncoderType.CLIP_G:
                self._encoders[encoder_type] = CLIPEncoder(use_projection=True)
            elif encoder_type == TextEncoderType.T5_XXL:
                self._encoders[encoder_type] = T5Encoder()
            elif encoder_type == TextEncoderType.QWEN3_KLEIN:
                self._encoders[encoder_type] = Qwen3Encoder(mode="klein")
            elif encoder_type == TextEncoderType.QWEN3_ZIMAGE:
                self._encoders[encoder_type] = Qwen3Encoder(mode="zimage")
            elif encoder_type == TextEncoderType.QWEN25_VL:
                self._encoders[encoder_type] = Qwen25VLEncoder(mode="txt2img")
            elif encoder_type == TextEncoderType.MISTRAL:
                self._encoders[encoder_type] = MistralEncoder()
            elif encoder_type == TextEncoderType.UMT5_XXL:
                self._encoders[encoder_type] = UMT5Encoder()
            else:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")
        return self._encoders[encoder_type]

    def load_encoder(
        self,
        encoder_type: TextEncoderType,
        model_path: str,
        dtype: Any = None,
        device: str = "cpu",
        subfolder: str | None = None,
        tokenizer_subfolder: str | None = None,
    ) -> None:
        """Load a specific encoder by type with an explicit path.

        Args:
            encoder_type: Which encoder to load.
            model_path: HuggingFace repo ID or local directory.
            dtype: Torch dtype for model weights.
            device: Target device string.
            subfolder: Optional subfolder for model weights.
            tokenizer_subfolder: Optional subfolder for tokenizer.
        """
        import torch

        if dtype is None:
            dtype = torch.float16

        encoder = self.get_encoder(encoder_type)
        if not encoder.is_loaded:
            encoder._dtype = dtype
            encoder._device = device
            if isinstance(encoder, CLIPEncoder):
                encoder.load(
                    model_path,
                    subfolder=subfolder,
                    tokenizer_subfolder=tokenizer_subfolder,
                )
            else:
                encoder.load(model_path)
            logger.info("Loaded %s from %s", encoder_type.value, model_path)

    # -- lifecycle -----------------------------------------------------------

    def load_for_model(
        self,
        model_architecture: ModelArchitecture,
        dtype: Any = None,
        device: str = "cpu",
    ) -> None:
        """Load all text encoders required by *model_architecture*.

        Uses :func:`get_default_encoder_path` to resolve HuggingFace
        repos for each encoder type.  For Klein and ZImage, the encoder
        path must be provided via :meth:`load_encoder` since they are
        bundled with the model checkpoint.

        Args:
            model_architecture: Target model architecture.
            dtype: Torch dtype for model weights.
            device: Target device string (e.g. ``"cuda"``).
        """
        import torch

        if dtype is None:
            dtype = torch.float16

        required = get_required_encoders(model_architecture)
        for enc_type in required:
            enc_path = get_default_encoder_path(model_architecture, enc_type)
            if enc_path is None:
                logger.warning(
                    "No default path for %s / %s -- load manually via load_encoder()",
                    model_architecture.value,
                    enc_type.value,
                )
                continue
            self.load_encoder(
                enc_type,
                model_path=enc_path.repo,
                dtype=dtype,
                device=device,
                subfolder=enc_path.subfolder,
                tokenizer_subfolder=enc_path.tokenizer_subfolder,
            )
            logger.info(
                "Loaded %s encoder for %s",
                enc_type.value,
                model_architecture.value,
            )

    def unload_all(self) -> None:
        """Unload and discard all cached encoders."""
        for encoder in self._encoders.values():
            encoder.unload()
        self._encoders.clear()

    # -- model-specific encoding ---------------------------------------------

    def encode_for_model(
        self,
        model_architecture: ModelArchitecture,
        prompt: str,
        negative: str = "",
        clip_skip: int = 0,
    ) -> dict[str, Any]:
        """Encode a prompt (and optional negative) for a specific model.

        Returns a dict of tensors whose keys depend on the model type.

        Raises:
            ValueError: If the model architecture is not supported.
            RuntimeError: If required encoders are not loaded.
        """
        dispatch = {
            ModelArchitecture.SD15: self._encode_sd15,
            ModelArchitecture.SD20: self._encode_sd15,
            ModelArchitecture.SD21: self._encode_sd15,
            ModelArchitecture.SDXL: self._encode_sdxl,
            ModelArchitecture.SDXL_REFINER: self._encode_sdxl_refiner,
            ModelArchitecture.SD3: self._encode_sd3,
            ModelArchitecture.FLUX_DEV: self._encode_flux1,
            ModelArchitecture.FLUX_FILL: self._encode_flux1,
            ModelArchitecture.FLUX_SCHNELL: self._encode_flux1,
            ModelArchitecture.FLUX_2_DEV: self._encode_flux2_dev,
            ModelArchitecture.FLUX_2_KLEIN_4B: self._encode_klein,
            ModelArchitecture.FLUX_2_KLEIN_9B: self._encode_klein,
            ModelArchitecture.CHROMA: self._encode_chroma,
            ModelArchitecture.WAN: self._encode_wan,
            ModelArchitecture.QWEN: self._encode_qwen,
            ModelArchitecture.LUMINA: self._encode_zimage,
            ModelArchitecture.ZIMAGE: self._encode_zimage,
            ModelArchitecture.PIXART_ALPHA: self._encode_wan,
            ModelArchitecture.PIXART_SIGMA: self._encode_wan,
            ModelArchitecture.SANA: self._encode_wan,
            ModelArchitecture.HUNYUAN_VIDEO: self._encode_wan,
            ModelArchitecture.WUERSTCHEN: self._encode_sd15,
            ModelArchitecture.STABLE_CASCADE: self._encode_sd15,
            ModelArchitecture.HIDREAM: self._encode_wan,
            ModelArchitecture.OMNIGEN: self._encode_sd15,
            ModelArchitecture.KANDINSKY: self._encode_sd15,
        }
        fn = dispatch.get(model_architecture)
        if fn is None:
            raise ValueError(f"Encoding not implemented for {model_architecture}")
        return fn(prompt, negative, clip_skip)

    # -- per-model encoders --------------------------------------------------

    def _encode_sd15(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """SD 1.5: CLIP-L only."""
        clip_l = self.get_encoder(TextEncoderType.CLIP_L)
        cond = clip_l.encode(prompt, clip_skip=clip_skip)
        uncond = clip_l.encode(negative, clip_skip=clip_skip)
        return {
            "cond": cond.hidden_states,
            "uncond": uncond.hidden_states,
        }

    def _encode_sdxl(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """SDXL: CLIP-L + CLIP-G, concatenated hidden states."""
        import torch

        clip_l = self.get_encoder(TextEncoderType.CLIP_L)
        clip_g = self.get_encoder(TextEncoderType.CLIP_G)

        cond_l = clip_l.encode(prompt, clip_skip=clip_skip)
        cond_g = clip_g.encode(prompt, clip_skip=clip_skip)
        uncond_l = clip_l.encode(negative, clip_skip=clip_skip)
        uncond_g = clip_g.encode(negative, clip_skip=clip_skip)

        max_len = max(cond_l.hidden_states.shape[1], cond_g.hidden_states.shape[1])
        cond_l_h = _pad_to_length(cond_l.hidden_states, max_len)
        cond_g_h = _pad_to_length(cond_g.hidden_states, max_len)
        uncond_l_h = _pad_to_length(uncond_l.hidden_states, max_len)
        uncond_g_h = _pad_to_length(uncond_g.hidden_states, max_len)

        return {
            "cond": torch.cat([cond_l_h, cond_g_h], dim=2),
            "uncond": torch.cat([uncond_l_h, uncond_g_h], dim=2),
            "pooled": cond_g.pooled_output,
            "neg_pooled": uncond_g.pooled_output,
        }

    def _encode_sdxl_refiner(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """SDXL Refiner: CLIP-G only."""
        clip_g = self.get_encoder(TextEncoderType.CLIP_G)
        cond = clip_g.encode(prompt, clip_skip=clip_skip)
        uncond = clip_g.encode(negative, clip_skip=clip_skip)
        return {
            "cond": cond.hidden_states,
            "uncond": uncond.hidden_states,
            "pooled": cond.pooled_output,
            "neg_pooled": uncond.pooled_output,
        }

    def _encode_flux1(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """Flux 1: CLIP-L (pooled) + T5-XXL (hidden states)."""
        clip_l = self.get_encoder(TextEncoderType.CLIP_L)
        t5 = self.get_encoder(TextEncoderType.T5_XXL)

        clip_out = clip_l.encode(prompt, clip_skip=clip_skip)
        t5_out = t5.encode(prompt)

        return {
            "cond": t5_out.hidden_states,
            "uncond": None,
            "clip_cond": clip_out.hidden_states,
            "pooled": clip_out.pooled_output,
        }

    def _encode_flux2_dev(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """Flux 2 Dev: Mistral3 24B, stacked layers [10,20,30], dim=15360."""
        mistral = self.get_encoder(TextEncoderType.MISTRAL)
        mistral_out = mistral.encode(prompt)
        return {
            "cond": mistral_out.hidden_states,
            "uncond": None,
        }

    def _encode_klein(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """Flux 2 Klein: Qwen3 mode=klein, stacked layers [9,18,27].

        Output shape:
        - Klein 4B: (B, seq, 7680) = 3 * 2560
        - Klein 9B: (B, seq, 12288) = 3 * 4096
        """
        qwen3 = self.get_encoder(TextEncoderType.QWEN3_KLEIN)
        qwen_out = qwen3.encode(prompt)
        return {
            "cond": qwen_out.hidden_states,
            "uncond": None,
        }

    def _encode_chroma(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """Chroma: T5-XXL only (distilled Flux variant)."""
        t5 = self.get_encoder(TextEncoderType.T5_XXL)
        t5_out = t5.encode(prompt)
        return {
            "cond": t5_out.hidden_states,
            "uncond": None,
        }

    def _encode_wan(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """Wan: UMT5-XXL (different vocab/tokenizer from standard T5-XXL)."""
        umt5 = self.get_encoder(TextEncoderType.UMT5_XXL)
        umt5_out, _mask = umt5.encode(prompt)
        return {
            "cond": umt5_out.hidden_states,
            "uncond": None,
        }

    def _encode_zimage(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """ZImage: Qwen3 mode=zimage, penultimate layer, dim=2560."""
        qwen3 = self.get_encoder(TextEncoderType.QWEN3_ZIMAGE)
        qwen_out = qwen3.encode(prompt)
        return {
            "cond": qwen_out.hidden_states,
            "uncond": None,
        }

    def _encode_qwen(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """Qwen Image: Qwen2.5-VL text encoder with attention masks."""
        encoder = self.get_encoder(TextEncoderType.QWEN25_VL)
        cond_out, cond_mask = encoder.encode(prompt)
        uncond_out, uncond_mask = encoder.encode(negative)
        return {
            "cond": cond_out.hidden_states,
            "uncond": uncond_out.hidden_states,
            "attention_mask": cond_mask,
            "uncond_attention_mask": uncond_mask,
        }

    def _encode_sd3(
        self,
        prompt: str,
        negative: str,
        clip_skip: int,
    ) -> dict[str, Any]:
        """SD3: CLIP-L + CLIP-G + T5-XXL, all concatenated."""
        import torch

        clip_l = self.get_encoder(TextEncoderType.CLIP_L)
        clip_g = self.get_encoder(TextEncoderType.CLIP_G)
        t5 = self.get_encoder(TextEncoderType.T5_XXL)

        cond_l = clip_l.encode(prompt, clip_skip=clip_skip)
        cond_g = clip_g.encode(prompt, clip_skip=clip_skip)
        t5_out = t5.encode(prompt)

        uncond_l = clip_l.encode(negative, clip_skip=clip_skip)
        uncond_g = clip_g.encode(negative, clip_skip=clip_skip)
        t5_uncond = t5.encode(negative)

        max_len = max(
            cond_l.hidden_states.shape[1],
            cond_g.hidden_states.shape[1],
            t5_out.hidden_states.shape[1],
        )
        cond_all = torch.cat([
            _pad_to_length(cond_l.hidden_states, max_len),
            _pad_to_length(cond_g.hidden_states, max_len),
            _pad_to_length(t5_out.hidden_states, max_len),
        ], dim=2)
        uncond_all = torch.cat([
            _pad_to_length(uncond_l.hidden_states, max_len),
            _pad_to_length(uncond_g.hidden_states, max_len),
            _pad_to_length(t5_uncond.hidden_states, max_len),
        ], dim=2)

        return {
            "cond": cond_all,
            "uncond": uncond_all,
            "pooled": cond_g.pooled_output,
            "neg_pooled": uncond_g.pooled_output,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_length(tensor: Any, target_len: int) -> Any:
    """Pad a ``(batch, seq, dim)`` tensor along the sequence axis."""
    import torch

    if tensor.shape[1] >= target_len:
        return tensor[:, :target_len, :]
    pad_size = target_len - tensor.shape[1]
    padding = torch.zeros(
        tensor.shape[0], pad_size, tensor.shape[2],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=1)
