"""Video nodes -- empty latent for video models, save, preview, combine, split."""
from __future__ import annotations

import logging
import os

import torch

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import wrap_latent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Empty latents (existing)
# ---------------------------------------------------------------------------

@registry.register(
    "EmptyLTXVLatentVideo",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "width": ("INT",), "height": ("INT",),
        "length": ("INT",), "batch_size": ("INT",),
    }},
)
def empty_ltxv_latent(width, height, length=25, batch_size=1):
    # LTX-Video: 128 channels, spatial //32, temporal //8
    latent = torch.zeros(batch_size, 128, (length + 7) // 8, height // 32, width // 32)
    return (wrap_latent(latent),)


@registry.register(
    "EmptyWanLatentVideo",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "width": ("INT",), "height": ("INT",),
        "length": ("INT",), "batch_size": ("INT",),
    }},
)
def empty_wan_latent(width, height, length=25, batch_size=1):
    # WAN: 16 channels, spatial //8, temporal is model-specific
    latent = torch.zeros(batch_size, 16, length, height // 8, width // 8)
    return (wrap_latent(latent),)


@registry.register(
    "EmptyHunyuanLatentVideo",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "width": ("INT",), "height": ("INT",),
        "length": ("INT",), "batch_size": ("INT",),
    }},
)
def empty_hunyuan_latent(width, height, length=25, batch_size=1):
    # HunyuanVideo: 16 channels, spatial //8
    latent = torch.zeros(batch_size, 16, length, height // 8, width // 8)
    return (wrap_latent(latent),)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

@registry.register(
    "VideoSave",
    return_types=(),
    category="video",
    is_output=True,
    input_types={"required": {
        "images": ("IMAGE",),
        "filename_prefix": ("STRING",),
        "fps": ("FLOAT",),
        "format": ("STRING",),
    }},
)
def video_save(images, filename_prefix="SerenityFlow", fps=24.0, format="mp4"):
    import subprocess
    import tempfile
    import numpy as np
    from PIL import Image

    from serenityflow.bridge.model_paths import get_model_paths
    paths = get_model_paths()
    output_dir = os.path.join(paths.base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Write frames to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(images.shape[0]):
            img_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_np).save(os.path.join(tmpdir, f"frame_{i:06d}.png"))

        output_path = os.path.join(output_dir, f"{filename_prefix}.{format}")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        log.info("Saved video: %s", output_path)

    return {"ui": {"videos": [{"filename": f"{filename_prefix}.{format}", "type": "output"}]}}


@registry.register(
    "VideoPreview",
    return_types=(),
    category="video",
    is_output=True,
    input_types={"required": {
        "images": ("IMAGE",),
        "fps": ("FLOAT",),
    }},
)
def video_preview(images, fps=24.0):
    # Save as temp gif for preview
    import numpy as np
    from PIL import Image

    from serenityflow.bridge.model_paths import get_model_paths
    paths = get_model_paths()
    temp_dir = os.path.join(paths.base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    frames = []
    for i in range(images.shape[0]):
        img_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np))

    filename = f"preview_{id(images)}.gif"
    filepath = os.path.join(temp_dir, filename)
    if frames:
        frames[0].save(filepath, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)

    return {"ui": {"videos": [{"filename": filename, "type": "temp"}]}}


# ---------------------------------------------------------------------------
# Frame manipulation
# ---------------------------------------------------------------------------

@registry.register(
    "VideoCombine",
    return_types=("IMAGE",),
    category="video",
    input_types={"required": {
        "images1": ("IMAGE",),
        "images2": ("IMAGE",),
    }},
)
def video_combine(images1, images2):
    # Concatenate frame sequences along batch dim
    return (torch.cat([images1, images2], dim=0),)


@registry.register(
    "SplitVideoFrames",
    return_types=("IMAGE", "IMAGE"),
    return_names=("first_half", "second_half"),
    category="video",
    input_types={"required": {
        "images": ("IMAGE",),
        "split_at": ("INT",),
    }},
)
def split_video_frames(images, split_at):
    split_at = max(0, min(split_at, images.shape[0]))
    return (images[:split_at], images[split_at:])


@registry.register(
    "MergeVideoFrames",
    return_types=("IMAGE",),
    category="video",
    input_types={"required": {
        "frames_a": ("IMAGE",),
        "frames_b": ("IMAGE",),
        "overlap": ("INT",),
        "blend_mode": ("STRING",),
    }},
)
def merge_video_frames(frames_a, frames_b, overlap=0, blend_mode="cut"):
    if overlap <= 0 or blend_mode == "cut":
        return (torch.cat([frames_a, frames_b], dim=0),)
    # Cross-fade overlap region
    oa = frames_a[-overlap:]
    ob = frames_b[:overlap]
    weights = torch.linspace(1, 0, overlap, device=frames_a.device).view(-1, 1, 1, 1)
    blended = oa * weights + ob * (1 - weights)
    return (torch.cat([frames_a[:-overlap], blended, frames_b[overlap:]], dim=0),)


@registry.register(
    "ImageToVideo",
    return_types=("IMAGE",),
    category="video",
    input_types={"required": {
        "image": ("IMAGE",),
        "num_frames": ("INT",),
    }},
)
def image_to_video(image, num_frames=25):
    # Repeat a single image into a video sequence
    if image.shape[0] == 1:
        return (image.repeat(num_frames, 1, 1, 1),)
    return (image,)


@registry.register(
    "VideoToFrames",
    return_types=("IMAGE",),
    return_names=("frames",),
    category="video",
    input_types={"required": {
        "images": ("IMAGE",),
        "start_frame": ("INT",),
        "end_frame": ("INT",),
    }},
)
def video_to_frames(images, start_frame=0, end_frame=-1):
    if end_frame < 0:
        end_frame = images.shape[0]
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, images.shape[0])
    return (images[start_frame:end_frame],)


# ---------------------------------------------------------------------------
# Additional video nodes
# ---------------------------------------------------------------------------

@registry.register(
    "SaveVideo",
    return_types=(),
    category="video",
    is_output=True,
    input_types={"required": {
        "images": ("IMAGE",),
        "filename_prefix": ("STRING",),
        "fps": ("FLOAT",),
        "format": ("STRING",),
    }},
)
def save_video(images, filename_prefix="SerenityFlow", fps=24.0, format="mp4"):
    return video_save(images, filename_prefix=filename_prefix, fps=fps, format=format)


@registry.register(
    "CreateVideo",
    return_types=("VIDEO",),
    category="video",
    input_types={"required": {
        "images": ("IMAGE",),
        "fps": ("FLOAT",),
    }},
)
def create_video(images, fps=24.0):
    return ({"frames": images, "fps": fps},)


@registry.register(
    "LoadVideo",
    return_types=("IMAGE", "FLOAT"),
    return_names=("frames", "fps"),
    category="video",
    input_types={"required": {
        "video": ("STRING",),
    }},
)
def load_video(video):
    raise NotImplementedError("LoadVideo requires ffmpeg integration")


@registry.register(
    "GetVideoComponents",
    return_types=("IMAGE", "FLOAT"),
    return_names=("frames", "fps"),
    category="video",
    input_types={"required": {
        "video": ("VIDEO",),
    }},
)
def get_video_components(video):
    if isinstance(video, dict) and "frames" in video:
        return (video["frames"], video.get("fps", 24.0))
    raise ValueError("Expected a VIDEO dict with 'frames' key")


@registry.register(
    "TrimVideoLatent",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",),
        "trim_start": ("INT",),
        "trim_end": ("INT",),
    }},
)
def trim_video_latent(samples, trim_start=0, trim_end=0):
    from serenityflow.bridge.types import unwrap_latent

    latent = unwrap_latent(samples)
    if latent.ndim == 5:
        # 5D: (B, C, T, H, W) -- trim along temporal dim 2
        t = latent.shape[2]
        end = t - trim_end if trim_end > 0 else t
        latent = latent[:, :, trim_start:end, :, :]
    return (wrap_latent(latent),)


@registry.register(
    "SaveAnimatedWEBP",
    return_types=(),
    category="video",
    is_output=True,
    input_types={"required": {
        "images": ("IMAGE",),
        "filename_prefix": ("STRING",),
        "fps": ("FLOAT",),
    }},
)
def save_animated_webp(images, filename_prefix="SerenityFlow", fps=24.0):
    import numpy as np
    from PIL import Image

    from serenityflow.bridge.model_paths import get_model_paths
    paths = get_model_paths()
    output_dir = os.path.join(paths.base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    for i in range(images.shape[0]):
        img_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np))

    filename = f"{filename_prefix}.webp"
    filepath = os.path.join(output_dir, filename)
    if frames:
        frames[0].save(
            filepath, save_all=True, append_images=frames[1:],
            duration=int(1000 / fps), loop=0, format="WEBP",
        )
        log.info("Saved animated WebP: %s", filepath)

    return {"ui": {"videos": [{"filename": filename, "type": "output"}]}}
