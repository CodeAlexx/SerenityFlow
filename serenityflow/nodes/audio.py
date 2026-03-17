"""Audio nodes -- load, save, encode, crop."""
from __future__ import annotations

import os

import torch

from serenityflow.nodes.registry import registry


def materialize_audio_path(audio) -> str:
    """Return a filesystem path for an AUDIO value, writing a temp WAV if needed."""
    import tempfile
    import wave

    if isinstance(audio, dict):
        path = audio.get("path")
        if path and os.path.exists(path):
            return str(path)
        waveform = audio.get("waveform")
        sample_rate = audio.get("sampling_rate") or audio.get("sample_rate")
    else:
        path = getattr(audio, "path", None)
        if path and os.path.exists(path):
            return str(path)
        waveform = getattr(audio, "waveform", None)
        sample_rate = getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None)

    if waveform is None or sample_rate is None:
        raise ValueError("AUDIO value does not contain a valid path or waveform")

    samples = waveform.detach().cpu().float()
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    if samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
        samples = samples.transpose(0, 1)
    if samples.ndim != 2:
        raise ValueError(f"Unsupported audio waveform shape: {list(samples.shape)}")
    if samples.shape[0] > 8:
        samples = samples.transpose(0, 1)

    interleaved = samples.transpose(0, 1).contiguous()
    pcm16 = interleaved.clamp(-1.0, 1.0).mul(32767.0).to(torch.int16).numpy()

    temp_dir = os.path.realpath("temp")
    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="sf_audio_", suffix=".wav", dir=temp_dir, delete=False) as handle:
        with wave.open(handle.name, "wb") as wav_file:
            wav_file.setnchannels(int(interleaved.shape[1]))
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm16.tobytes())
        return handle.name


def _save_audio_encoded(audio, filename_prefix: str, extension: str, codec: str):
    import subprocess

    source_path = materialize_audio_path(audio)
    output_dir = os.path.realpath("output")
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"{filename_prefix}.{extension}"
    output_path = os.path.join(output_dir, output_name)
    cmd = ["ffmpeg", "-y", "-i", source_path, "-vn", "-c:a", codec, output_path]
    subprocess.run(cmd, capture_output=True, check=True)
    return {"ui": {"audio": [{"filename": output_name, "type": "output"}]}}


@registry.register(
    "LoadAudio",
    return_types=("AUDIO",),
    category="audio",
    input_types={"required": {"audio": ("STRING",)}},
)
def load_audio(audio):
    from serenityflow.bridge.model_paths import get_model_paths

    if os.path.isabs(audio):
        filepath = audio
    else:
        paths = get_model_paths()
        input_dir = os.path.join(paths.base_dir, "input")
        candidate = os.path.join(input_dir, audio)
        if os.path.exists(candidate):
            filepath = candidate
        elif os.path.exists(audio):
            filepath = os.path.realpath(audio)
        else:
            filepath = candidate

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio not found: {filepath}")

    return ({"path": filepath, "waveform": None, "sample_rate": None},)


@registry.register(
    "SaveAudioMP3",
    return_types=(),
    category="audio",
    is_output=True,
    input_types={"required": {"audio": ("AUDIO",), "filename_prefix": ("STRING",)}},
)
def save_audio_mp3(audio, filename_prefix="serenityflow"):
    return _save_audio_encoded(audio, filename_prefix, "mp3", "libmp3lame")


@registry.register(
    "SaveAudioOpus",
    return_types=(),
    category="audio",
    is_output=True,
    input_types={"required": {"audio": ("AUDIO",), "filename_prefix": ("STRING",)}},
)
def save_audio_opus(audio, filename_prefix="serenityflow"):
    return _save_audio_encoded(audio, filename_prefix, "opus", "libopus")


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
