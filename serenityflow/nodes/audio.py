"""Audio nodes -- load, save, encode, crop."""
from __future__ import annotations

from serenityflow.nodes.registry import registry


@registry.register(
    "LoadAudio",
    return_types=("AUDIO",),
    category="audio",
    input_types={"required": {"audio": ("STRING",)}},
)
def load_audio(audio):
    return ({"path": audio, "waveform": None, "sample_rate": None},)


@registry.register(
    "SaveAudioMP3",
    return_types=(),
    category="audio",
    is_output=True,
    input_types={"required": {"audio": ("AUDIO",), "filename_prefix": ("STRING",)}},
)
def save_audio_mp3(audio, filename_prefix="serenityflow"):
    raise NotImplementedError("SaveAudioMP3 requires audio encoding backend")


@registry.register(
    "SaveAudioOpus",
    return_types=(),
    category="audio",
    is_output=True,
    input_types={"required": {"audio": ("AUDIO",), "filename_prefix": ("STRING",)}},
)
def save_audio_opus(audio, filename_prefix="serenityflow"):
    raise NotImplementedError("SaveAudioOpus requires audio encoding backend")


@registry.register(
    "RecordAudio",
    return_types=("AUDIO",),
    category="audio",
    input_types={"required": {}},
)
def record_audio():
    raise NotImplementedError("RecordAudio requires audio capture device")


@registry.register(
    "VAEDecodeAudio",
    return_types=("AUDIO",),
    category="audio",
    input_types={"required": {"samples": ("LATENT",), "vae": ("VAE",)}},
)
def vae_decode_audio(samples, vae):
    raise NotImplementedError("VAEDecodeAudio requires audio VAE backend")


@registry.register(
    "AudioCrop",
    return_types=("AUDIO",),
    category="audio",
    input_types={"required": {
        "audio": ("AUDIO",),
        "start_time": ("FLOAT",),
        "end_time": ("FLOAT",),
    }},
)
def audio_crop(audio, start_time, end_time):
    out = dict(audio) if isinstance(audio, dict) else {"path": None, "waveform": None, "sample_rate": None}
    out["crop_start"] = start_time
    out["crop_end"] = end_time
    return (out,)


@registry.register(
    "AudioEncoderEncode",
    return_types=("LATENT",),
    category="audio",
    input_types={"required": {
        "audio_encoder": ("AUDIO_ENCODER",),
        "audio": ("AUDIO",),
    }},
)
def audio_encoder_encode(audio_encoder, audio):
    raise NotImplementedError("AudioEncoderEncode requires audio encoder backend")


@registry.register(
    "AudioEncoderLoader",
    return_types=("AUDIO_ENCODER",),
    category="audio",
    input_types={"required": {"model_name": ("STRING",)}},
)
def audio_encoder_loader(model_name):
    raise NotImplementedError("AudioEncoderLoader requires audio encoder backend")


@registry.register(
    "TextEncodeAceStepAudio",
    return_types=("CONDITIONING",),
    category="conditioning/audio",
    input_types={"required": {
        "clip": ("CLIP",),
        "lyrics": ("STRING",),
        "tags": ("STRING",),
    }},
)
def text_encode_ace_step_audio(clip, lyrics, tags):
    raise NotImplementedError("TextEncodeAceStepAudio requires ACE-Step backend")


@registry.register(
    "TextEncodeAceStepAudio1.5",
    return_types=("CONDITIONING",),
    category="conditioning/audio",
    input_types={"required": {
        "clip": ("CLIP",),
        "lyrics": ("STRING",),
        "tags": ("STRING",),
    }},
)
def text_encode_ace_step_audio_1_5(clip, lyrics, tags):
    raise NotImplementedError("TextEncodeAceStepAudio1.5 requires ACE-Step 1.5 backend")


@registry.register(
    "EmptyAceStepLatentAudio",
    return_types=("LATENT",),
    category="latent/audio",
    input_types={"required": {
        "duration": ("FLOAT",),
        "batch_size": ("INT",),
    }},
)
def empty_ace_step_latent_audio(duration, batch_size=1):
    return ({"samples": None, "duration": duration, "batch_size": batch_size},)


@registry.register(
    "EmptyAceStep1.5LatentAudio",
    return_types=("LATENT",),
    category="latent/audio",
    input_types={"required": {
        "duration": ("FLOAT",),
        "batch_size": ("INT",),
    }},
)
def empty_ace_step_1_5_latent_audio(duration, batch_size=1):
    return ({"samples": None, "duration": duration, "batch_size": batch_size},)


@registry.register(
    "LatentApplyOperationCFG",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "operation": ("LATENT_OPERATION",),
    }},
)
def latent_apply_operation_cfg(model, operation):
    if hasattr(model, "with_options"):
        return (model.with_options({"latent_operation_cfg": operation}),)
    return (model,)


@registry.register(
    "LatentOperationTonemapReinhard",
    return_types=("LATENT_OPERATION",),
    category="latent/operation",
    input_types={"required": {}},
)
def latent_operation_tonemap_reinhard():
    return ({"type": "tonemap_reinhard"},)
