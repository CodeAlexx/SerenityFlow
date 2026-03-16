"""Video pipeline nodes -- I/O, latent ops, temporal VAE, interpolation, AnimateDiff, Wan/LTX additions."""
from __future__ import annotations

import glob as glob_module
import logging
import os
import subprocess

import numpy as np
import torch

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import unwrap_latent, wrap_latent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


@registry.register(
    "LoadVideo",
    return_types=("IMAGE", "INT", "FLOAT"),
    return_names=("frames", "frame_count", "fps"),
    category="video",
    input_types={"required": {
        "video_path": ("STRING",),
    },
    "optional": {
        "frame_limit": ("INT", {"default": 0, "min": 0}),
        "skip_first_frames": ("INT", {"default": 0, "min": 0}),
        "select_every_nth": ("INT", {"default": 1, "min": 1}),
        "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0}),
    }},
)
def load_video(video_path, frame_limit=0, skip_first_frames=0,
               select_every_nth=1, force_rate=0.0):
    """Load video file to IMAGE batch [N,H,W,C] float32 0-1."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames_list = []
    fps = 24.0

    # Try imageio first
    try:
        import imageio.v3 as iio
        meta = iio.immeta(video_path, plugin="pyav")
        fps = meta.get("fps", 24.0)
        all_frames = iio.imread(video_path, plugin="pyav")
        for i, frame in enumerate(all_frames):
            if i < skip_first_frames:
                continue
            if (i - skip_first_frames) % select_every_nth != 0:
                continue
            frames_list.append(frame)
            if frame_limit > 0 and len(frames_list) >= frame_limit:
                break
    except (ImportError, Exception):
        try:
            import imageio
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            fps = meta.get("fps", 24.0)
            for i, frame in enumerate(reader):
                if i < skip_first_frames:
                    continue
                if (i - skip_first_frames) % select_every_nth != 0:
                    continue
                frames_list.append(frame)
                if frame_limit > 0 and len(frames_list) >= frame_limit:
                    break
            reader.close()
        except (ImportError, Exception):
            # Fallback: PIL for GIFs
            from PIL import Image
            img = Image.open(video_path)
            idx = 0
            try:
                while True:
                    if idx >= skip_first_frames and (idx - skip_first_frames) % select_every_nth == 0:
                        frames_list.append(np.array(img.convert("RGB")))
                        if frame_limit > 0 and len(frames_list) >= frame_limit:
                            break
                    idx += 1
                    img.seek(idx)
            except EOFError:
                pass

    if not frames_list:
        raise RuntimeError(f"No frames extracted from: {video_path}")

    if force_rate > 0.0:
        fps = force_rate

    # Stack to [N,H,W,C] float32 0-1
    arr = np.stack(frames_list, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    frame_count = tensor.shape[0]
    log.info("LoadVideo: %d frames at %.1f fps from %s", frame_count, fps, video_path)
    return (tensor, frame_count, fps)


@registry.register(
    "LoadVideoFrames",
    return_types=("IMAGE", "INT"),
    return_names=("frames", "count"),
    category="video",
    input_types={"required": {
        "directory": ("STRING",),
    },
    "optional": {
        "pattern": ("STRING", {"default": "*.png"}),
    }},
)
def load_video_frames(directory, pattern="*.png"):
    """Load directory of frame images into IMAGE batch."""
    from PIL import Image

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(glob_module.glob(os.path.join(directory, pattern)))
    if not files:
        raise RuntimeError(f"No files matching '{pattern}' in {directory}")

    frames = []
    for f in files:
        img = Image.open(f).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        frames.append(torch.from_numpy(arr))

    batch = torch.stack(frames, dim=0)  # [N, H, W, C]
    log.info("LoadVideoFrames: %d frames from %s", batch.shape[0], directory)
    return (batch, batch.shape[0])


@registry.register(
    "SaveVideoFrames",
    return_types=(),
    category="video",
    is_output=True,
    input_types={"required": {
        "images": ("IMAGE",),
        "output_dir": ("STRING",),
    },
    "optional": {
        "prefix": ("STRING", {"default": "frame_"}),
    }},
)
def save_video_frames(images, output_dir, prefix="frame_"):
    """Save IMAGE batch as individual frame files."""
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    count = images.shape[0]
    for i in range(count):
        frame = images[i]  # [H, W, C]
        arr = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        path = os.path.join(output_dir, f"{prefix}{i:05d}.png")
        pil.save(path)

    log.info("SaveVideoFrames: saved %d frames to %s", count, output_dir)
    return {"ui": {"frame_count": count, "output_dir": output_dir}}


@registry.register(
    "CombineVideoFrames",
    return_types=("STRING",),
    return_names=("video_path",),
    category="video",
    input_types={"required": {
        "images": ("IMAGE",),
        "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
        "output_path": ("STRING",),
    },
    "optional": {
        "codec": (["h264", "h265", "vp9"], {"default": "h264"}),
        "quality": ("INT", {"default": 23, "min": 0, "max": 51}),
    }},
)
def combine_video_frames(images, fps=24.0, output_path="output.mp4",
                         codec="h264", quality=23):
    """Combine IMAGE batch into video file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_np = (images.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # Try imageio first
    try:
        import imageio
        codec_map = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9"}
        writer = imageio.get_writer(
            output_path, fps=fps, codec=codec_map.get(codec, "libx264"),
            quality=None, output_params=["-crf", str(quality)],
        )
        for i in range(frames_np.shape[0]):
            writer.append_data(frames_np[i])
        writer.close()
    except (ImportError, Exception):
        # Fallback: ffmpeg subprocess
        h, w = frames_np.shape[1], frames_np.shape[2]
        codec_map = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9"}
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-", "-c:v", codec_map.get(codec, "libx264"),
            "-crf", str(quality), "-pix_fmt", "yuv420p", output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for i in range(frames_np.shape[0]):
            proc.stdin.write(frames_np[i].tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode()}")

    log.info("CombineVideoFrames: wrote %d frames to %s", images.shape[0], output_path)
    return (output_path,)


@registry.register(
    "SplitVideoFrames",
    return_types=("IMAGE",),
    return_names=("first_frame",),
    category="video",
    input_types={"required": {"images": ("IMAGE",)}},
)
def split_video_frames(images):
    """Extract first frame from IMAGE batch."""
    return (images[0:1],)


@registry.register(
    "MergeVideoFrames",
    return_types=("IMAGE",),
    return_names=("merged",),
    category="video",
    input_types={"required": {
        "frames_1": ("IMAGE",),
        "frames_2": ("IMAGE",),
    }},
)
def merge_video_frames(frames_1, frames_2):
    """Concatenate two frame batches along batch dimension."""
    # Resize frames_2 to match frames_1 spatial dims if needed
    if frames_1.shape[1:3] != frames_2.shape[1:3]:
        import torch.nn.functional as F
        f2 = frames_2.permute(0, 3, 1, 2)
        f2 = F.interpolate(f2, size=frames_1.shape[1:3], mode="bilinear", align_corners=False)
        frames_2 = f2.permute(0, 2, 3, 1)
    return (torch.cat([frames_1, frames_2], dim=0),)


@registry.register(
    "GetVideoInfo",
    return_types=("INT", "FLOAT", "INT", "INT", "FLOAT"),
    return_names=("frame_count", "fps", "width", "height", "duration"),
    category="video",
    input_types={"required": {"video_path": ("STRING",)}},
)
def get_video_info(video_path):
    """Extract video metadata."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Try ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                w = int(stream.get("width", 0))
                h = int(stream.get("height", 0))
                nb_frames = int(stream.get("nb_frames", 0))
                r_parts = stream.get("r_frame_rate", "24/1").split("/")
                fps = float(r_parts[0]) / float(r_parts[1]) if len(r_parts) == 2 else 24.0
                duration = float(data.get("format", {}).get("duration", 0.0))
                return (nb_frames, fps, w, h, duration)
    except (FileNotFoundError, subprocess.CalledProcessError, Exception):
        pass

    # Fallback: imageio
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = meta.get("fps", 24.0)
        w = meta.get("size", (0, 0))[0]
        h = meta.get("size", (0, 0))[1]
        n = reader.count_frames()
        duration = n / fps if fps > 0 else 0.0
        reader.close()
        return (n, fps, w, h, duration)
    except (ImportError, Exception):
        raise RuntimeError(f"Cannot read video info from {video_path}")


# ---------------------------------------------------------------------------
# AnimateDiff
# ---------------------------------------------------------------------------


@registry.register(
    "AnimateDiffLoader",
    return_types=("ANIMATEDIFF_MODEL",),
    return_names=("animatediff_model",),
    category="loaders/video",
    input_types={"required": {"model_name": ("STRING",)}},
)
def animatediff_loader(model_name):
    """Load AnimateDiff motion module. Returns handle dict for bridge."""
    log.info("AnimateDiffLoader: %s", model_name)
    return ({"model_name": model_name, "type": "animatediff"},)


@registry.register(
    "AnimateDiffSettings",
    return_types=("ANIMATEDIFF_SETTINGS",),
    return_names=("settings",),
    category="video/animatediff",
    input_types={"required": {
        "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
        "beta_schedule": (["linear", "sqrt_linear", "cosine"], {"default": "linear"}),
    }},
)
def animatediff_settings(motion_scale=1.0, beta_schedule="linear"):
    """Configure AnimateDiff motion parameters."""
    return ({"motion_scale": motion_scale, "beta_schedule": beta_schedule},)


@registry.register(
    "AnimateDiffCombine",
    return_types=("IMAGE",),
    return_names=("frames",),
    category="video/animatediff",
    input_types={"required": {
        "images": ("IMAGE",),
    },
    "optional": {
        "interpolation": (["none", "film"], {"default": "none"}),
        "loop": ("BOOLEAN", {"default": False}),
    }},
)
def animatediff_combine(images, interpolation="none", loop=False):
    """Post-process AnimateDiff video frames."""
    result = images
    if loop:
        # Append reversed frames (excluding first and last to avoid stutter)
        if result.shape[0] > 2:
            reversed_frames = result[1:-1].flip(0)
            result = torch.cat([result, reversed_frames], dim=0)
    # interpolation='film' would need FILM model — pass through for now
    return (result,)


# ---------------------------------------------------------------------------
# Video Latent Operations
# ---------------------------------------------------------------------------


@registry.register(
    "SetLatentBatchSize",
    return_types=("LATENT",),
    return_names=("samples",),
    category="latent/video",
    input_types={"required": {
        "samples": ("LATENT",),
        "batch_size": ("INT", {"default": 1, "min": 1}),
    }},
)
def set_latent_batch_size(samples, batch_size=1):
    """Resize latent batch for target frame count. Repeats or truncates."""
    latent = unwrap_latent(samples)
    current = latent.shape[0]
    if batch_size == current:
        return (wrap_latent(latent),)
    elif batch_size < current:
        return (wrap_latent(latent[:batch_size]),)
    else:
        # Repeat to fill, then truncate
        repeats = (batch_size + current - 1) // current
        expanded = latent.repeat(repeats, *([1] * (latent.ndim - 1)))
        return (wrap_latent(expanded[:batch_size]),)


@registry.register(
    "LatentBatchSlice",
    return_types=("LATENT",),
    return_names=("samples",),
    category="latent/video",
    input_types={"required": {
        "samples": ("LATENT",),
        "start": ("INT", {"default": 0, "min": 0}),
        "end": ("INT", {"default": -1}),
    }},
)
def latent_batch_slice(samples, start=0, end=-1):
    """Extract frame range from latent batch."""
    latent = unwrap_latent(samples)
    if end < 0:
        end = latent.shape[0]
    result = latent[start:end]
    return (wrap_latent(result.contiguous()),)


# ---------------------------------------------------------------------------
# Wan 2.2 Specific
# ---------------------------------------------------------------------------


@registry.register(
    "WanT2V",
    return_types=("LATENT",),
    return_names=("latent",),
    category="sampling/video",
    input_types={"required": {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "frames": ("INT", {"default": 81, "min": 1, "max": 512}),
        "width": ("INT", {"default": 832, "min": 64, "max": 1920, "step": 16}),
        "height": ("INT", {"default": 480, "min": 64, "max": 1088, "step": 16}),
        "seed": ("INT", {"default": 0}),
        "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
        "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0}),
    }},
)
def wan_t2v(model, positive, negative, frames=81, width=832, height=480,
            seed=0, steps=30, cfg=5.0):
    """Wan 2.2 text-to-video sampling. Delegates to bridge."""
    config = {
        "type": "wan_t2v",
        "model": model,
        "positive": positive,
        "negative": negative,
        "frames": frames,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
    }
    # Bridge would call the actual model here
    raise NotImplementedError("WanT2V requires bridge video sampling")


@registry.register(
    "WanV2V",
    return_types=("LATENT",),
    return_names=("latent",),
    category="sampling/video",
    input_types={"required": {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "video": ("IMAGE",),
        "seed": ("INT", {"default": 0}),
        "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
        "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0}),
    },
    "optional": {
        "denoise": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
    }},
)
def wan_v2v(model, positive, negative, video, seed=0, steps=30, cfg=5.0,
            denoise=0.7):
    """Wan 2.2 video-to-video style transfer. Delegates to bridge."""
    config = {
        "type": "wan_v2v",
        "model": model,
        "positive": positive,
        "negative": negative,
        "video": video,
        "denoise": denoise,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
    }
    raise NotImplementedError("WanV2V requires bridge video sampling")


# ---------------------------------------------------------------------------
# LTX-V Additions
# ---------------------------------------------------------------------------


@registry.register(
    "LTXAudioSync",
    return_types=("AUDIO",),
    return_names=("audio",),
    category="sampling/video",
    input_types={"required": {
        "video_latent": ("LATENT",),
        "audio_conditioning": ("CONDITIONING",),
    }},
)
def ltx_audio_sync(video_latent, audio_conditioning):
    """LTX-AV 2.3 audio synchronization. Returns AUDIO dict."""
    config = {
        "type": "ltx_audio_sync",
        "video_latent": video_latent,
        "audio_conditioning": audio_conditioning,
    }
    raise NotImplementedError("LTXAudioSync requires bridge audio-video sync")


# ---------------------------------------------------------------------------
# Temporal VAE
# ---------------------------------------------------------------------------


@registry.register(
    "VAEDecodeVideo",
    return_types=("IMAGE",),
    return_names=("frames",),
    category="latent/video",
    input_types={"required": {
        "vae": ("VAE",),
        "samples": ("LATENT",),
    },
    "optional": {
        "tile_temporal": ("INT", {"default": 8, "min": 1, "max": 256}),
    }},
)
def vae_decode_video(vae, samples, tile_temporal=8):
    """Temporal VAE decode for video latents. Decodes in chunks to avoid OOM."""
    from serenityflow.bridge.types import bchw_to_bhwc

    latent = unwrap_latent(samples)
    total = latent.shape[0]
    decoded_chunks = []

    for start in range(0, total, tile_temporal):
        end = min(start + tile_temporal, total)
        chunk = latent[start:end]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # VAE decode: expects BCHW, returns BCHW
        if hasattr(vae, "decode"):
            decoded = vae.decode(chunk)
            if hasattr(decoded, "sample"):
                decoded = decoded.sample
        else:
            from serenityflow.bridge.serenity_api import vae_decode
            decoded = vae_decode(vae, chunk)

        decoded_chunks.append(decoded)

    result = torch.cat(decoded_chunks, dim=0)
    # Convert BCHW -> BHWC
    result = bchw_to_bhwc(result)
    log.info("VAEDecodeVideo: decoded %d frames in chunks of %d", total, tile_temporal)
    return (result,)


@registry.register(
    "VAEEncodeVideo",
    return_types=("LATENT",),
    return_names=("samples",),
    category="latent/video",
    input_types={"required": {
        "vae": ("VAE",),
        "images": ("IMAGE",),
    },
    "optional": {
        "tile_temporal": ("INT", {"default": 8, "min": 1, "max": 256}),
    }},
)
def vae_encode_video(vae, images, tile_temporal=8):
    """Temporal VAE encode for video frames. Encodes in chunks to avoid OOM."""
    from serenityflow.bridge.types import bhwc_to_bchw

    pixels = bhwc_to_bchw(images)  # BHWC -> BCHW
    total = pixels.shape[0]
    encoded_chunks = []

    for start in range(0, total, tile_temporal):
        end = min(start + tile_temporal, total)
        chunk = pixels[start:end]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(vae, "encode"):
            encoded = vae.encode(chunk)
            if hasattr(encoded, "latent_dist"):
                encoded = encoded.latent_dist.sample()
            elif hasattr(encoded, "sample"):
                encoded = encoded.sample
        else:
            from serenityflow.bridge.serenity_api import vae_encode
            encoded = vae_encode(vae, chunk)

        encoded_chunks.append(encoded)

    result = torch.cat(encoded_chunks, dim=0)
    log.info("VAEEncodeVideo: encoded %d frames in chunks of %d", total, tile_temporal)
    return (wrap_latent(result),)


# ---------------------------------------------------------------------------
# Frame Interpolation (RIFE)
# ---------------------------------------------------------------------------


@registry.register(
    "LoadRIFEModel",
    return_types=("RIFE_MODEL",),
    return_names=("rife_model",),
    category="loaders/video",
    input_types={"required": {"model_name": ("STRING",)}},
)
def load_rife_model(model_name):
    """Load RIFE interpolation model. Returns handle dict for bridge."""
    log.info("LoadRIFEModel: %s", model_name)
    return ({"model_name": model_name, "type": "rife"},)


@registry.register(
    "RIFEInterpolate",
    return_types=("IMAGE",),
    return_names=("frames",),
    category="video",
    input_types={"required": {
        "rife_model": ("RIFE_MODEL",),
        "images": ("IMAGE",),
        "multiplier": ("INT", {"default": 2, "min": 2, "max": 8}),
    }},
)
def rife_interpolate(rife_model, images, multiplier=2):
    """Interpolate between frames using RIFE or linear fallback.

    For N input frames with multiplier M, produces N + (N-1)*(M-1) output frames.
    """
    n = images.shape[0]
    if n < 2:
        return (images,)

    has_rife = False
    rife_fn = None

    # Try to use actual RIFE model if bridge provides it
    if isinstance(rife_model, dict) and "interpolate_fn" in rife_model:
        rife_fn = rife_model["interpolate_fn"]
        has_rife = True

    output_frames = [images[0:1]]

    for i in range(n - 1):
        frame_a = images[i:i + 1]  # [1, H, W, C]
        frame_b = images[i + 1:i + 2]

        for step in range(1, multiplier):
            t = step / multiplier
            if has_rife and rife_fn is not None:
                interp = rife_fn(frame_a, frame_b, t)
            else:
                # Linear interpolation fallback
                interp = frame_a * (1.0 - t) + frame_b * t
            output_frames.append(interp)

        output_frames.append(frame_b)

    result = torch.cat(output_frames, dim=0)
    log.info("RIFEInterpolate: %d -> %d frames (x%d)", n, result.shape[0], multiplier)
    return (result,)
