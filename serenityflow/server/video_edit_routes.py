"""Video Edit project CRUD endpoints."""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, Response

log = logging.getLogger(__name__)


def _projects_dir() -> str:
    from serenityflow.server.app import state
    d = os.path.join(os.path.realpath(state.output_dir), "video_projects")
    os.makedirs(d, exist_ok=True)
    return d


def _project_path(project_id: str) -> str | None:
    base = _projects_dir()
    path = os.path.realpath(os.path.join(base, f"{project_id}.json"))
    if not path.startswith(os.path.realpath(base) + os.sep):
        return None
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_video_edit_routes(app: FastAPI):

    @app.get("/video_edit/projects")
    async def list_video_projects():
        base = _projects_dir()
        projects = []
        for fname in sorted(os.listdir(base)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(base, fname), "r") as f:
                    proj = json.load(f)
                projects.append({
                    "id": proj.get("id", fname[:-5]),
                    "name": proj.get("name", "Untitled"),
                    "updated_at": proj.get("updated_at"),
                })
            except Exception:
                continue
        return JSONResponse(projects)

    @app.post("/video_edit/projects")
    async def create_video_project(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}

        project_id = body.get("id") or str(uuid.uuid4())[:12]
        now = _now_iso()
        proj = {
            "id": project_id,
            "name": body.get("name", "Untitled Project"),
            "fps": body.get("fps", 30),
            "width": body.get("width", 1280),
            "height": body.get("height", 720),
            "tracks": body.get("tracks", []),
            "created_at": now,
            "updated_at": now,
        }

        path = _project_path(project_id)
        if not path:
            return JSONResponse({"error": "Invalid project ID"}, status_code=400)

        with open(path, "w") as f:
            json.dump(proj, f, indent=2)

        return JSONResponse(proj, status_code=201)

    @app.get("/video_edit/projects/{project_id}")
    async def get_video_project(project_id: str):
        path = _project_path(project_id)
        if not path or not os.path.isfile(path):
            return JSONResponse({"error": "Not found"}, status_code=404)

        with open(path, "r") as f:
            proj = json.load(f)
        return JSONResponse(proj)

    @app.put("/video_edit/projects/{project_id}")
    async def update_video_project(project_id: str, request: Request):
        path = _project_path(project_id)
        if not path:
            return JSONResponse({"error": "Invalid project ID"}, status_code=400)

        body = await request.json()
        body["id"] = project_id
        body["updated_at"] = _now_iso()

        # Preserve created_at if file exists
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    old = json.load(f)
                body.setdefault("created_at", old.get("created_at"))
            except Exception:
                pass

        with open(path, "w") as f:
            json.dump(body, f, indent=2)

        return JSONResponse(body)

    @app.delete("/video_edit/projects/{project_id}")
    async def delete_video_project(project_id: str):
        path = _project_path(project_id)
        if not path or not os.path.isfile(path):
            return JSONResponse({"error": "Not found"}, status_code=404)

        os.remove(path)
        # Also remove media dir if exists
        media_dir = os.path.join(_projects_dir(), project_id, "media")
        if os.path.isdir(media_dir):
            shutil.rmtree(os.path.join(_projects_dir(), project_id), ignore_errors=True)

        return JSONResponse({"status": "deleted"})

    @app.post("/video_edit/projects/{project_id}/import_clip")
    async def import_clip(project_id: str, file: UploadFile = File(...)):
        path = _project_path(project_id)
        if not path:
            return JSONResponse({"error": "Invalid project ID"}, status_code=400)

        # Create media dir
        media_dir = os.path.join(_projects_dir(), project_id, "media")
        os.makedirs(media_dir, exist_ok=True)

        clip_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(file.filename or "clip.mp4")[1]
        dest = os.path.join(media_dir, f"{clip_id}{ext}")

        content = await file.read()
        if len(content) > 500 * 1024 * 1024:  # 500MB
            return JSONResponse({"error": "File too large"}, status_code=413)

        with open(dest, "wb") as f:
            f.write(content)

        # Probe with ffprobe
        info = {"clip_id": clip_id, "source_path": dest}
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", dest,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            probe = json.loads(result.stdout) if result.stdout else {}

            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["width"] = int(stream.get("width", 0))
                    info["height"] = int(stream.get("height", 0))
                    # Calculate duration in frames
                    fps_str = stream.get("r_frame_rate", "30/1")
                    try:
                        num, den = fps_str.split("/")
                        info["fps"] = round(int(num) / int(den))
                    except Exception:
                        info["fps"] = 30
                    nb_frames = stream.get("nb_frames")
                    if nb_frames and nb_frames != "N/A":
                        info["duration_frames"] = int(nb_frames)
                    else:
                        dur = float(probe.get("format", {}).get("duration", 0))
                        info["duration_frames"] = int(dur * info["fps"])
                    break
                elif stream.get("codec_type") == "audio" and "duration_frames" not in info:
                    dur = float(stream.get("duration", probe.get("format", {}).get("duration", 0)))
                    info["duration_frames"] = int(dur * 30)
                    info["fps"] = 30
        except Exception as e:
            log.warning("ffprobe failed for %s: %s", dest, e)
            info["duration_frames"] = 150  # default 5s

        return JSONResponse(info)

    # --- Thumbnail sprite extraction ---

    @app.post("/video_edit/thumbnails")
    async def extract_thumbnails(request: Request):
        body = await request.json()
        source = body.get("source_path", "")
        thumb_h = body.get("height", 36)

        if not source or not os.path.isfile(source):
            return JSONResponse({"error": "Source file not found"}, status_code=400)

        # Validate path is within projects dir
        base = _projects_dir()
        source_real = os.path.realpath(source)
        if not source_real.startswith(os.path.realpath(base)):
            return JSONResponse({"error": "Access denied"}, status_code=403)

        project_dir = os.path.dirname(source)
        basename = os.path.splitext(os.path.basename(source))[0]
        sprite_name = f"thumbs_{basename}_{thumb_h}.jpg"
        sprite_path = os.path.join(project_dir, sprite_name)

        if not os.path.exists(sprite_path):
            tmp_dir = os.path.join(project_dir, "_thumbs_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            try:
                # Extract 1 frame per second
                subprocess.run([
                    "ffmpeg", "-y", "-i", source,
                    "-vf", f"fps=1,scale=-1:{thumb_h}",
                    "-q:v", "8",
                    os.path.join(tmp_dir, "frame_%04d.jpg"),
                ], capture_output=True, timeout=120)

                frames = sorted(glob.glob(os.path.join(tmp_dir, "frame_*.jpg")))
                if frames:
                    from PIL import Image
                    imgs = [Image.open(f) for f in frames]
                    w = imgs[0].width
                    sprite = Image.new("RGB", (w * len(imgs), thumb_h))
                    for i, img in enumerate(imgs):
                        sprite.paste(img.resize((w, thumb_h)), (i * w, 0))
                    sprite.save(sprite_path, quality=70)
            except Exception as e:
                log.warning("Thumbnail extraction failed for %s: %s", source, e)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if not os.path.exists(sprite_path):
            return JSONResponse({"error": "Extraction failed"}, status_code=500)

        # Compute dimensions
        try:
            from PIL import Image
            img = Image.open(sprite_path)
            # Assume 16:9 aspect per thumb
            thumb_w = max(1, round(thumb_h * 16 / 9))
            frame_count = max(1, img.width // thumb_w)
        except Exception:
            thumb_w = round(thumb_h * 16 / 9)
            frame_count = 1

        # Return a serve-able URL
        rel_path = os.path.relpath(sprite_path, base)
        return JSONResponse({
            "sprite_url": "/video_edit/media/" + rel_path.replace(os.sep, "/"),
            "frame_count": frame_count,
            "thumb_width": thumb_w,
            "thumb_height": thumb_h,
        })

    # --- Waveform extraction ---

    @app.post("/video_edit/waveform")
    async def extract_waveform(request: Request):
        body = await request.json()
        source = body.get("source_path", "")
        sps = body.get("samples_per_second", 30)

        if not source or not os.path.isfile(source):
            return JSONResponse({"error": "Source file not found"}, status_code=400)

        base = _projects_dir()
        source_real = os.path.realpath(source)
        if not source_real.startswith(os.path.realpath(base)):
            return JSONResponse({"error": "Access denied"}, status_code=403)

        # Check disk cache
        cache_path = source + ".waveform.json"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return JSONResponse(json.load(f))
            except Exception:
                pass

        # Extract raw PCM mono audio
        oversample = sps * 100
        try:
            cmd = [
                "ffmpeg", "-y", "-i", source,
                "-ac", "1", "-ar", str(oversample),
                "-f", "f32le", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            raw = result.stdout
        except Exception as e:
            log.warning("Waveform extraction failed for %s: %s", source, e)
            return JSONResponse({"peaks": [], "sample_rate": sps, "duration_seconds": 0})

        if not raw:
            return JSONResponse({"peaks": [], "sample_rate": sps, "duration_seconds": 0})

        import struct
        sample_count = len(raw) // 4
        samples = struct.unpack(f"<{sample_count}f", raw)

        duration = sample_count / oversample
        total_peaks = max(1, int(duration * sps))
        chunk_size = max(1, sample_count // total_peaks)

        peaks = []
        for i in range(0, sample_count, chunk_size):
            chunk = samples[i : i + chunk_size]
            peaks.append(max(abs(s) for s in chunk) if chunk else 0.0)

        # Normalize
        max_peak = max(peaks) if peaks else 1.0
        if max_peak > 0:
            peaks = [p / max_peak for p in peaks]

        result_data = {
            "peaks": peaks,
            "sample_rate": sps,
            "duration_seconds": round(duration, 3),
        }

        # Cache to disk
        try:
            with open(cache_path, "w") as f:
                json.dump(result_data, f)
        except Exception:
            pass

        return JSONResponse(result_data)

    # --- Frame extraction helper ---

    def _extract_frame(source: str, time_sec: float, output: str) -> bool:
        """Extract a single frame from a video at the given timestamp."""
        try:
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(max(0, time_sec)),
                "-i", source, "-frames:v", "1",
                "-q:v", "2", output,
            ], capture_output=True, timeout=30)
            return os.path.isfile(output)
        except Exception as e:
            log.warning("Frame extraction failed: %s", e)
            return False

    def _find_clip_in_project(proj: dict, clip_id: str) -> dict | None:
        for track in proj.get("tracks", []):
            for clip in track.get("clips", []):
                if clip.get("id") == clip_id:
                    return clip
        return None

    def _get_project_media_dir(project_id: str) -> str:
        d = os.path.join(_projects_dir(), project_id, "media")
        os.makedirs(d, exist_ok=True)
        return d

    # --- Retake endpoint ---

    @app.post("/video_edit/retake")
    async def retake_clip(request: Request):
        body = await request.json()
        pid = body.get("project_id", "")
        clip_id = body.get("clip_id", "")
        region_start = body.get("region_start_frame", 0)
        region_end = body.get("region_end_frame", 0)
        prompt = body.get("prompt", "")
        strength = body.get("strength", 0.7)
        seed = body.get("seed")

        path = _project_path(pid)
        if not path or not os.path.isfile(path):
            return JSONResponse({"error": "Project not found"}, status_code=404)

        with open(path, "r") as f:
            proj = json.load(f)

        clip = _find_clip_in_project(proj, clip_id)
        if not clip:
            return JSONResponse({"error": "Clip not found"}, status_code=404)

        source = clip.get("source_path", "")
        if not source or not os.path.isfile(source):
            return JSONResponse({"error": "Clip has no source file"}, status_code=400)

        fps = proj.get("fps", 30)
        source_start_sec = region_start / fps
        duration_sec = (region_end - region_start) / fps

        media_dir = _get_project_media_dir(pid)
        retakes_dir = os.path.join(media_dir, "retakes")
        os.makedirs(retakes_dir, exist_ok=True)

        take_id = str(uuid.uuid4())[:12]
        segment_path = os.path.join(retakes_dir, f"segment_{take_id}.mp4")
        output_path = os.path.join(retakes_dir, f"retake_{take_id}.mp4")

        # Extract the source segment
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", source,
                "-ss", str(source_start_sec),
                "-t", str(duration_sec),
                "-c:v", "libx264", "-crf", "18",
                segment_path,
            ], capture_output=True, timeout=120)
        except Exception as e:
            return JSONResponse({"error": f"Segment extraction failed: {e}"}, status_code=500)

        # Extract boundary frames
        frame_before_path = os.path.join(retakes_dir, f"before_{take_id}.jpg")
        frame_after_path = os.path.join(retakes_dir, f"after_{take_id}.jpg")
        _extract_frame(source, source_start_sec - 1.0 / fps, frame_before_path)
        _extract_frame(source, source_start_sec + duration_sec, frame_after_path)

        # Build a v2v workflow and queue it via /prompt
        # The workflow structure follows the existing ComfyUI-protocol pattern
        width = clip.get("width") or proj.get("width", 1280)
        height = clip.get("height") or proj.get("height", 720)
        frames_count = region_end - region_start

        workflow = {
            "client_id": "video_edit_retake",
            "prompt": {
                "1": {
                    "class_type": "SerenityRetake",
                    "inputs": {
                        "input_video": segment_path,
                        "prompt": prompt,
                        "strength": strength,
                        "width": width,
                        "height": height,
                        "frames": frames_count,
                        "seed": seed if seed is not None else -1,
                        "frame_before": frame_before_path if os.path.isfile(frame_before_path) else "",
                        "frame_after": frame_after_path if os.path.isfile(frame_after_path) else "",
                        "output_path": output_path,
                    },
                },
            },
        }

        # Try to queue — graceful failure if no model loaded
        prompt_id = None
        try:
            from serenityflow.server.app import state
            prompt_id = str(uuid.uuid4())
            await state.prompt_queue.put({
                "prompt_id": prompt_id,
                "prompt": workflow["prompt"],
                "client_id": workflow.get("client_id", ""),
                "extra_data": {},
            })
        except Exception as e:
            log.warning("Failed to queue retake: %s", e)
            return JSONResponse({
                "take_id": take_id,
                "status": "error",
                "error": str(e),
                "output_path": output_path,
            })

        return JSONResponse({
            "take_id": take_id,
            "status": "queued",
            "prompt_id": prompt_id,
            "output_path": output_path,
            "segment_path": segment_path,
        })

    # --- Bridge shot endpoint ---

    @app.post("/video_edit/bridge_shot")
    async def bridge_shot(request: Request):
        body = await request.json()
        pid = body.get("project_id", "")
        track_id = body.get("track_id", "")
        gap_start = body.get("gap_start_frame", 0)
        gap_end = body.get("gap_end_frame", 0)
        before_clip_id = body.get("before_clip_id")
        after_clip_id = body.get("after_clip_id")
        prompt = body.get("prompt", "")
        seed = body.get("seed")

        path = _project_path(pid)
        if not path or not os.path.isfile(path):
            return JSONResponse({"error": "Project not found"}, status_code=404)

        with open(path, "r") as f:
            proj = json.load(f)

        fps = proj.get("fps", 30)
        media_dir = _get_project_media_dir(pid)
        bridges_dir = os.path.join(media_dir, "bridges")
        os.makedirs(bridges_dir, exist_ok=True)

        bridge_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(bridges_dir, f"bridge_{bridge_id}.mp4")

        # Extract boundary frames
        start_img = os.path.join(bridges_dir, f"start_{bridge_id}.jpg")
        end_img = os.path.join(bridges_dir, f"end_{bridge_id}.jpg")

        if before_clip_id:
            bc = _find_clip_in_project(proj, before_clip_id)
            if bc and bc.get("source_path") and os.path.isfile(bc["source_path"]):
                # Compute source-relative time for the last frame of the clip
                bc_sf = bc.get("startFrame", bc.get("start_frame", 0))
                bc_ef = bc.get("endFrame", bc.get("end_frame", 0))
                bc_src_start = (bc.get("source_start", 0) or 0)
                last_frame_sec = (bc_src_start + bc_ef - bc_sf - 1) / fps
                _extract_frame(bc["source_path"], last_frame_sec, start_img)

        if after_clip_id:
            ac = _find_clip_in_project(proj, after_clip_id)
            if ac and ac.get("source_path") and os.path.isfile(ac["source_path"]):
                first_frame_sec = (ac.get("source_start", 0) or 0) / fps
                _extract_frame(ac["source_path"], first_frame_sec, end_img)

        duration_frames = gap_end - gap_start
        width = proj.get("width", 1280)
        height = proj.get("height", 720)

        # Try local prompt enhancement if prompt is short
        enhanced_prompt = prompt
        if len(prompt.split()) < 10:
            try:
                import httpx
                resp = httpx.post(
                    f"http://127.0.0.1:8188/enhance_prompt",
                    json={"prompt": prompt or "smooth cinematic transition"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    enhanced_prompt = data.get("enhanced", prompt) or prompt
            except Exception:
                pass  # Use original prompt

        # Build i2v workflow
        workflow = {
            "client_id": "video_edit_bridge",
            "prompt": {
                "1": {
                    "class_type": "SerenityBridgeShot",
                    "inputs": {
                        "start_image": start_img if os.path.isfile(start_img) else "",
                        "end_image": end_img if os.path.isfile(end_img) else "",
                        "prompt": enhanced_prompt,
                        "frames": duration_frames,
                        "width": width,
                        "height": height,
                        "seed": seed if seed is not None else -1,
                        "output_path": output_path,
                    },
                },
            },
        }

        # Queue
        prompt_id = None
        try:
            from serenityflow.server.app import state
            prompt_id = str(uuid.uuid4())
            await state.prompt_queue.put({
                "prompt_id": prompt_id,
                "prompt": workflow["prompt"],
                "client_id": workflow.get("client_id", ""),
                "extra_data": {},
            })
        except Exception as e:
            log.warning("Failed to queue bridge shot: %s", e)
            return JSONResponse({
                "clip_id": "bridge_" + bridge_id,
                "status": "error",
                "error": str(e),
            })

        # Add the clip to project immediately (placeholder — source will be filled on completion)
        new_clip = {
            "id": "bridge_" + bridge_id,
            "startFrame": gap_start,
            "endFrame": gap_end,
            "label": "Bridge Shot",
            "color": "#ff8c42",
            "source_path": output_path,
            "generating": True,
        }

        for track in proj.get("tracks", []):
            if track.get("id") == track_id:
                track["clips"].append(new_clip)
                track["clips"].sort(key=lambda c: c.get("startFrame", c.get("start_frame", 0)))
                break

        proj["updated_at"] = _now_iso()
        with open(path, "w") as f:
            json.dump(proj, f, indent=2)

        return JSONResponse({
            "clip_id": new_clip["id"],
            "status": "queued",
            "prompt_id": prompt_id,
            "output_path": output_path,
        })

    # --- Resolve /view URL to filesystem path ---

    @app.post("/video_edit/resolve_view_path")
    async def resolve_view_path(request: Request):
        """Resolve a /view?filename=...&type=... URL to a real filesystem path."""
        body = await request.json()
        filename = body.get("filename", "")
        subfolder = body.get("subfolder", "")
        ftype = body.get("type", "output")

        from serenityflow.server.app import state
        if ftype == "output":
            base = os.path.realpath(state.output_dir)
        elif ftype == "input":
            base = os.path.realpath(state.input_dir)
        elif ftype == "temp":
            base = os.path.realpath(state.temp_dir)
        else:
            return JSONResponse({"error": "Invalid type"}, status_code=400)

        filepath = os.path.realpath(os.path.join(base, subfolder, filename))
        if not filepath.startswith(base + os.sep) and filepath != base:
            return JSONResponse({"error": "Access denied"}, status_code=403)
        if not os.path.isfile(filepath):
            return JSONResponse({"error": "File not found"}, status_code=404)

        return JSONResponse({"path": filepath})

    # --- Static file serving for project media & thumbnails ---

    @app.get("/video_edit/media/{path:path}")
    async def serve_video_edit_media(path: str):
        base = _projects_dir()
        full = os.path.realpath(os.path.join(base, path))
        if not full.startswith(os.path.realpath(base) + os.sep):
            return JSONResponse({"error": "Access denied"}, status_code=403)
        if not os.path.isfile(full):
            return JSONResponse({"error": "Not found"}, status_code=404)
        return FileResponse(full)

    # --- V5: Video Export ---

    _active_exports: dict[str, subprocess.Popen] = {}

    def _get_codec_args(fmt: str, quality: str) -> list[str]:
        crf_map = {"low": "28", "medium": "23", "high": "18", "lossless": "0"}
        crf = crf_map.get(quality, "23")
        if fmt == "h264":
            return ["-c:v", "libx264", "-crf", crf, "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k"]
        elif fmt == "prores":
            profile = "3" if quality == "lossless" else "2"
            return ["-c:v", "prores_ks", "-profile:v", profile, "-c:a", "pcm_s16le"]
        elif fmt == "vp9":
            return ["-c:v", "libvpx-vp9", "-crf", crf, "-b:v", "0",
                    "-c:a", "libopus", "-b:a", "192k"]
        return ["-c:v", "libx264", "-crf", crf]

    @app.post("/video_edit/export")
    async def export_video(request: Request):
        body = await request.json()
        pid = body.get("project_id", "")
        fmt = body.get("format", "h264")
        width = body.get("width", 1280)
        height = body.get("height", 720)
        fps = body.get("fps", 30)
        quality = body.get("quality", "high")
        include_audio = body.get("include_audio", True)
        range_start = body.get("range_start_frame")
        range_end = body.get("range_end_frame")
        output_filename = body.get("output_filename", f"export_{pid}.mp4")

        path = _project_path(pid)
        if not path or not os.path.isfile(path):
            return JSONResponse({"error": "Project not found"}, status_code=404)

        with open(path, "r") as f:
            proj = json.load(f)

        export_id = str(uuid.uuid4())[:12]

        # Sanitize output_filename to prevent path traversal
        output_filename = os.path.basename(output_filename)
        if not output_filename:
            output_filename = f"export_{pid}.mp4"

        # Capture event loop for WS events from background thread
        loop = asyncio.get_running_loop()

        # Run export in background
        loop.run_in_executor(
            None, _run_export, proj, export_id, fmt, width, height, fps,
            quality, include_audio, range_start, range_end, output_filename, loop
        )

        return JSONResponse({"export_id": export_id, "status": "started"})

    def _run_export(proj, export_id, fmt, width, height, fps, quality,
                    include_audio, range_start, range_end, output_filename, loop=None):
        from serenityflow.server.app import state

        tracks = proj.get("tracks", [])
        proj_fps = proj.get("fps", 30)

        # Determine range
        if range_start is None:
            range_start = 0
        if range_end is None:
            max_f = 0
            for t in tracks:
                for c in t.get("clips", []):
                    ef = c.get("endFrame", c.get("end_frame", 0))
                    if ef > max_f:
                        max_f = ef
            range_end = max_f or 300

        duration_sec = (range_end - range_start) / proj_fps
        if duration_sec <= 0:
            _send_export_event(state, export_id, "export_error", {"error": "Zero duration"}, loop)
            return

        export_dir = os.path.join(os.path.realpath(state.output_dir), "exports")
        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.join(export_dir, output_filename)

        # Collect video clips in render order (track index 0 = bottom)
        video_inputs = []
        audio_inputs = []
        for track in tracks:
            ttype = track.get("type", "video")
            for clip in track.get("clips", []):
                sf = clip.get("startFrame", clip.get("start_frame", 0))
                ef = clip.get("endFrame", clip.get("end_frame", 0))
                if ef <= range_start or sf >= range_end:
                    continue
                src = clip.get("source_path", "")
                if not src or not os.path.isfile(src):
                    continue
                clip_start = max(sf, range_start)
                clip_end = min(ef, range_end)
                source_offset = (clip.get("source_start", 0) or 0) / proj_fps
                timeline_offset = (clip_start - range_start) / proj_fps
                clip_dur = (clip_end - clip_start) / proj_fps
                entry = {
                    "src": src, "source_offset": source_offset,
                    "timeline_offset": timeline_offset, "duration": clip_dur,
                }
                if ttype == "video":
                    video_inputs.append(entry)
                elif ttype == "audio":
                    audio_inputs.append(entry)

        # Build ffmpeg command using concat demuxer approach with overlay
        # Simpler: render each video clip individually, then concat with black fill
        # For correctness: use filter_complex with overlay chain
        try:
            cmd = _build_export_cmd(
                video_inputs, audio_inputs, width, height, fps,
                duration_sec, fmt, quality, include_audio, output_path
            )

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            _active_exports[export_id] = process

            total_frames = int(duration_sec * fps)
            last_stderr_lines = []

            # Parse ffmpeg stderr for progress
            for line_bytes in iter(process.stderr.readline, b""):
                line = line_bytes.decode("utf-8", errors="replace")
                last_stderr_lines.append(line)
                if len(last_stderr_lines) > 20:
                    last_stderr_lines.pop(0)
                match = re.search(r"frame=\s*(\d+)", line)
                if match:
                    frame = int(match.group(1))
                    pct = min(100, int(frame / max(1, total_frames) * 100))
                    _send_export_event(state, export_id, "export_progress", {
                        "frame": frame, "total_frames": total_frames, "percent": pct,
                    }, loop)

            process.wait()
            _active_exports.pop(export_id, None)

            if process.returncode == 0:
                _send_export_event(state, export_id, "export_complete", {
                    "output_path": output_path,
                }, loop)
            else:
                stderr_tail = "".join(last_stderr_lines)[-500:]
                _send_export_event(state, export_id, "export_error", {
                    "error": stderr_tail or "ffmpeg exited with code " + str(process.returncode),
                }, loop)
        except Exception as e:
            _active_exports.pop(export_id, None)
            _send_export_event(state, export_id, "export_error", {"error": str(e)}, loop)

    def _build_export_cmd(video_inputs, audio_inputs, width, height, fps,
                          duration, fmt, quality, include_audio, output_path):
        """Build ffmpeg command with filter_complex for multi-track compositing."""
        cmd = ["ffmpeg", "-y"]
        filter_parts = []
        input_idx = 0

        # Black background input
        cmd += ["-f", "lavfi", "-i",
                f"color=c=black:s={width}x{height}:d={duration}:r={fps}"]
        bg_label = f"[{input_idx}:v]"
        input_idx += 1

        # Add all video inputs
        v_labels = []
        for vi in video_inputs:
            cmd += ["-i", vi["src"]]
            label = f"v{input_idx}"
            filter_parts.append(
                f"[{input_idx}:v]"
                f"trim=start={vi['source_offset']:.4f}:duration={vi['duration']:.4f},"
                f"setpts=PTS-STARTPTS,"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                f"setpts=PTS+{vi['timeline_offset']:.4f}/TB"
                f"[{label}]"
            )
            v_labels.append((label, vi["timeline_offset"]))
            input_idx += 1

        # Add audio inputs
        a_labels = []
        if include_audio:
            for ai in audio_inputs:
                cmd += ["-i", ai["src"]]
                label = f"a{input_idx}"
                delay_ms = int(ai["timeline_offset"] * 1000)
                filter_parts.append(
                    f"[{input_idx}:a]"
                    f"atrim=start={ai['source_offset']:.4f}:duration={ai['duration']:.4f},"
                    f"asetpts=PTS-STARTPTS,"
                    f"adelay={delay_ms}|{delay_ms}"
                    f"[{label}]"
                )
                a_labels.append(label)
                input_idx += 1

        # Build overlay chain
        current = "base"
        filter_parts.insert(0, f"{bg_label}copy[base]")
        for i, (vl, _) in enumerate(v_labels):
            out = f"tmp{i}" if i < len(v_labels) - 1 else "vout"
            filter_parts.append(
                f"[{current}][{vl}]overlay=eof_action=pass:shortest=0[{out}]"
            )
            current = out

        if not v_labels:
            filter_parts.append(f"[base]copy[vout]")

        # Audio mix
        if a_labels and include_audio:
            amix_in = "".join(f"[{al}]" for al in a_labels)
            filter_parts.append(f"{amix_in}amix=inputs={len(a_labels)}:dropout_transition=0[aout]")
            map_args = ["-map", "[vout]", "-map", "[aout]"]
        else:
            map_args = ["-map", "[vout]"]

        cmd += ["-filter_complex", ";".join(filter_parts)]
        cmd += map_args
        cmd += ["-t", f"{duration:.4f}"]
        cmd += _get_codec_args(fmt, quality)
        cmd += [output_path]
        return cmd

    def _send_export_event(state, export_id, event_type, data, loop=None):
        """Send export event via WebSocket (sync — called from thread)."""
        data["export_id"] = export_id
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                _async_send_export(state, event_type, data), loop
            )
        except Exception:
            pass

    async def _async_send_export(state, event_type, data):
        from serenityflow.server.websocket import send_event
        await send_event(state, event_type, data)

    @app.post("/video_edit/export/{export_id}/cancel")
    async def cancel_export(export_id: str):
        proc = _active_exports.pop(export_id, None)
        if proc and proc.poll() is None:
            proc.kill()
            return JSONResponse({"status": "cancelled"})
        return JSONResponse({"status": "not_found"}, status_code=404)

    # --- V5: SRT Import/Export ---

    @app.post("/video_edit/projects/{project_id}/import_srt")
    async def import_srt(project_id: str, file: UploadFile = File(...)):
        proj_path = _project_path(project_id)
        if not proj_path or not os.path.isfile(proj_path):
            return JSONResponse({"error": "Project not found"}, status_code=404)

        content = (await file.read()).decode("utf-8", errors="replace")
        fps = 30

        with open(proj_path, "r") as f:
            proj = json.load(f)
        fps = proj.get("fps", 30)

        clips = _parse_srt(content, fps)
        if not clips:
            return JSONResponse({"error": "No subtitles found in SRT"}, status_code=400)

        track = {
            "id": f"srt_{uuid.uuid4().hex[:8]}",
            "name": "Subtitles",
            "type": "text",
            "clips": clips,
        }
        proj.setdefault("tracks", []).append(track)
        proj["updated_at"] = _now_iso()
        with open(proj_path, "w") as f:
            json.dump(proj, f, indent=2)

        return JSONResponse({"track_id": track["id"], "clip_count": len(clips)})

    def _parse_srt(content: str, fps: int) -> list[dict]:
        content = content.lstrip("\ufeff")  # Strip UTF-8 BOM
        blocks = re.split(r"\n\n+", content.strip())
        clips = []
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue
            time_match = re.match(
                r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
                lines[1],
            )
            if not time_match:
                continue
            g = time_match.groups()
            start_sec = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
            end_sec = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
            text = "\n".join(lines[2:])
            clips.append({
                "id": f"sub_{uuid.uuid4().hex[:8]}",
                "startFrame": round(start_sec * fps),
                "endFrame": round(end_sec * fps),
                "label": text,
                "color": "#d4a72c",
            })
        return clips

    @app.get("/video_edit/projects/{project_id}/export_srt")
    async def export_srt(project_id: str):
        proj_path = _project_path(project_id)
        if not proj_path or not os.path.isfile(proj_path):
            return JSONResponse({"error": "Not found"}, status_code=404)

        with open(proj_path, "r") as f:
            proj = json.load(f)

        fps = proj.get("fps", 30)
        subs = []
        for track in proj.get("tracks", []):
            if track.get("type") != "text":
                continue
            for clip in track.get("clips", []):
                subs.append(clip)
        subs.sort(key=lambda c: c.get("startFrame", c.get("start_frame", 0)))

        lines = []
        for idx, sub in enumerate(subs, 1):
            sf = sub.get("startFrame", sub.get("start_frame", 0))
            ef = sub.get("endFrame", sub.get("end_frame", 0))
            lines.append(str(idx))
            lines.append(f"{_frame_to_srt_time(sf, fps)} --> {_frame_to_srt_time(ef, fps)}")
            lines.append(sub.get("label", ""))
            lines.append("")

        srt_content = "\n".join(lines)
        return Response(
            content=srt_content,
            media_type="text/srt",
            headers={"Content-Disposition": f"attachment; filename=subtitles_{project_id}.srt"},
        )

    def _frame_to_srt_time(frame: int, fps: int) -> str:
        total_ms = round(frame / fps * 1000)
        h = total_ms // 3600000
        m = (total_ms % 3600000) // 60000
        s = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # --- V5: XML Timeline Import/Export ---

    @app.post("/video_edit/projects/{project_id}/import_xml")
    async def import_xml(project_id: str, file: UploadFile = File(...)):
        proj_path = _project_path(project_id)
        if not proj_path or not os.path.isfile(proj_path):
            return JSONResponse({"error": "Project not found"}, status_code=404)

        content = (await file.read()).decode("utf-8", errors="replace")
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            return JSONResponse({"error": f"Invalid XML: {e}"}, status_code=400)

        warnings = []

        # Detect format
        if root.tag == "fcpxml":
            tracks, warnings = _parse_fcpxml(root)
        elif root.find(".//sequence") is not None:
            tracks, warnings = _parse_premiere_xml(root)
        else:
            return JSONResponse({"error": "Unrecognized XML timeline format"}, status_code=400)

        with open(proj_path, "r") as f:
            proj = json.load(f)

        proj.setdefault("tracks", []).extend(tracks)
        proj["updated_at"] = _now_iso()
        with open(proj_path, "w") as f:
            json.dump(proj, f, indent=2)

        return JSONResponse({"tracks_imported": len(tracks), "warnings": warnings})

    def _parse_fcpxml_time(t: str, default_fps: int = 30) -> int:
        """Parse FCP XML time strings like '10s', '100/2400s', '3600/24000s'."""
        if not t:
            return 0
        try:
            t = t.strip().rstrip("s")
            if "/" in t:
                num, den = t.split("/")
                den_f = float(den)
                if den_f == 0:
                    return 0
                return round(float(num) / den_f * default_fps)
            return round(float(t) * default_fps)
        except (ValueError, ZeroDivisionError):
            return 0

    def _parse_fcpxml(root) -> tuple[list[dict], list[str]]:
        tracks = []
        warnings = []
        fps = 30

        # Try to get fps from resources
        for fmt in root.iter("format"):
            dur = fmt.get("frameDuration", "")
            if "/" in dur.rstrip("s"):
                try:
                    num, den = dur.rstrip("s").split("/")
                    fps = round(float(den) / float(num))
                except Exception:
                    pass

        for spine in root.iter("spine"):
            track = {
                "id": f"import_{uuid.uuid4().hex[:8]}",
                "name": "Imported Video",
                "type": "video",
                "clips": [],
            }
            offset = 0
            for elem in spine:
                duration = _parse_fcpxml_time(elem.get("duration", "0s"), fps)
                start = _parse_fcpxml_time(elem.get("start", "0s"), fps)
                name = elem.get("name", "Imported")

                # Try to resolve media ref
                ref_id = elem.get("ref", "")
                media_path = None
                if ref_id:
                    for asset in root.iter("asset"):
                        if asset.get("id") == ref_id:
                            src = asset.get("src", "")
                            if src.startswith("file://"):
                                media_path = src[7:]
                            elif src:
                                media_path = src
                            break
                    if not media_path:
                        warnings.append(f"Missing media for ref '{ref_id}'")

                track["clips"].append({
                    "id": f"imp_{uuid.uuid4().hex[:8]}",
                    "startFrame": offset,
                    "endFrame": offset + duration,
                    "label": name,
                    "color": "#6a9fd8",
                    "source_path": media_path,
                    "source_start": start,
                })
                offset += duration
            if track["clips"]:
                tracks.append(track)

        return tracks, warnings

    def _parse_premiere_xml(root) -> tuple[list[dict], list[str]]:
        tracks = []
        warnings = []

        for seq in root.iter("sequence"):
            fps_elem = seq.find(".//rate/timebase")
            fps = int(fps_elem.text) if fps_elem is not None and fps_elem.text else 30

            for v_track in seq.findall(".//video/track"):
                track = {
                    "id": f"import_{uuid.uuid4().hex[:8]}",
                    "name": "Imported Video",
                    "type": "video",
                    "clips": [],
                }
                for clip_elem in v_track.findall("clipitem"):
                    start_elem = clip_elem.find("start")
                    end_elem = clip_elem.find("end")
                    name_elem = clip_elem.find("name")
                    in_elem = clip_elem.find("in")

                    sf = int(start_elem.text) if start_elem is not None and start_elem.text else 0
                    ef = int(end_elem.text) if end_elem is not None and end_elem.text else sf + 150
                    name = name_elem.text if name_elem is not None else "Imported"
                    src_start = int(in_elem.text) if in_elem is not None and in_elem.text else 0

                    # Media path
                    file_elem = clip_elem.find(".//file/pathurl")
                    media_path = None
                    if file_elem is not None and file_elem.text:
                        p = file_elem.text
                        if p.startswith("file://localhost"):
                            media_path = p[16:]
                        elif p.startswith("file://"):
                            media_path = p[7:]
                        else:
                            media_path = p
                        if media_path and not os.path.isfile(media_path):
                            warnings.append(f"Missing media: {media_path}")
                            media_path = None

                    track["clips"].append({
                        "id": f"imp_{uuid.uuid4().hex[:8]}",
                        "startFrame": sf,
                        "endFrame": ef,
                        "label": name,
                        "color": "#6a9fd8",
                        "source_path": media_path,
                        "source_start": src_start,
                    })
                if track["clips"]:
                    tracks.append(track)

            for a_track in seq.findall(".//audio/track"):
                track = {
                    "id": f"import_{uuid.uuid4().hex[:8]}",
                    "name": "Imported Audio",
                    "type": "audio",
                    "clips": [],
                }
                for clip_elem in a_track.findall("clipitem"):
                    start_elem = clip_elem.find("start")
                    end_elem = clip_elem.find("end")
                    name_elem = clip_elem.find("name")
                    sf = int(start_elem.text) if start_elem is not None and start_elem.text else 0
                    ef = int(end_elem.text) if end_elem is not None and end_elem.text else sf + 150
                    name = name_elem.text if name_elem is not None else "Imported Audio"

                    track["clips"].append({
                        "id": f"imp_{uuid.uuid4().hex[:8]}",
                        "startFrame": sf,
                        "endFrame": ef,
                        "label": name,
                        "color": "#2a9d5c",
                    })
                if track["clips"]:
                    tracks.append(track)

        return tracks, warnings

    @app.get("/video_edit/projects/{project_id}/export_xml")
    async def export_xml(project_id: str, request: Request):
        fmt = request.query_params.get("format", "fcpxml")
        proj_path = _project_path(project_id)
        if not proj_path or not os.path.isfile(proj_path):
            return JSONResponse({"error": "Not found"}, status_code=404)

        with open(proj_path, "r") as f:
            proj = json.load(f)

        if fmt == "fcpxml":
            xml_str = _export_fcpxml(proj)
            filename = f"timeline_{project_id}.fcpxml"
        else:
            xml_str = _export_premiere_xml(proj)
            filename = f"timeline_{project_id}.xml"

        return Response(
            content=xml_str,
            media_type="application/xml",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    def _export_fcpxml(proj: dict) -> str:
        fps = proj.get("fps", 30)
        name = proj.get("name", "Untitled")
        root = ET.Element("fcpxml", version="1.9")

        resources = ET.SubElement(root, "resources")
        fmt = ET.SubElement(resources, "format",
                            id="r1", name=f"{proj.get('width', 1280)}x{proj.get('height', 720)}p{fps}",
                            frameDuration=f"100/{fps * 100}s",
                            width=str(proj.get("width", 1280)),
                            height=str(proj.get("height", 720)))

        library = ET.SubElement(root, "library")
        event = ET.SubElement(library, "event", name=name)
        project_el = ET.SubElement(event, "project", name=name)
        sequence = ET.SubElement(project_el, "sequence", format="r1",
                                 duration=f"{_frames_to_fcptime(proj, fps)}s")
        spine = ET.SubElement(sequence, "spine")

        asset_idx = 0
        for track in proj.get("tracks", []):
            if track.get("type") != "video":
                continue
            for clip in sorted(track.get("clips", []),
                               key=lambda c: c.get("startFrame", 0)):
                sf = clip.get("startFrame", 0)
                ef = clip.get("endFrame", 0)
                dur = ef - sf
                src = clip.get("source_path", "")

                # Register asset (always increment to avoid ID collisions)
                asset_id = f"a{asset_idx}"
                asset_idx += 1
                if src:
                    ET.SubElement(resources, "asset",
                                 id=asset_id, name=clip.get("label", ""),
                                 src=f"file://{src}")

                clip_el = ET.SubElement(spine, "asset-clip",
                                        ref=asset_id if src else "",
                                        name=clip.get("label", "Clip"),
                                        duration=f"{dur}/{fps}s",
                                        start=f"{clip.get('source_start', 0)}/{fps}s")

        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")

    def _export_premiere_xml(proj: dict) -> str:
        fps = proj.get("fps", 30)
        root = ET.Element("xmeml", version="5")
        seq = ET.SubElement(root, "sequence")
        ET.SubElement(seq, "name").text = proj.get("name", "Untitled")

        rate = ET.SubElement(seq, "rate")
        ET.SubElement(rate, "timebase").text = str(fps)
        ET.SubElement(rate, "ntsc").text = "FALSE"

        media = ET.SubElement(seq, "media")
        video = ET.SubElement(media, "video")
        audio = ET.SubElement(media, "audio")

        file_idx = 0
        for track in proj.get("tracks", []):
            ttype = track.get("type", "video")
            parent = video if ttype == "video" else audio if ttype == "audio" else None
            if parent is None:
                continue

            track_el = ET.SubElement(parent, "track")
            for clip in sorted(track.get("clips", []),
                               key=lambda c: c.get("startFrame", 0)):
                sf = clip.get("startFrame", 0)
                ef = clip.get("endFrame", 0)
                src = clip.get("source_path", "")

                ci = ET.SubElement(track_el, "clipitem", id=f"clipitem-{file_idx}")
                ET.SubElement(ci, "name").text = clip.get("label", "Clip")
                ET.SubElement(ci, "start").text = str(sf)
                ET.SubElement(ci, "end").text = str(ef)
                ET.SubElement(ci, "in").text = str(clip.get("source_start", 0))
                ET.SubElement(ci, "out").text = str(clip.get("source_start", 0) + ef - sf)

                ci_rate = ET.SubElement(ci, "rate")
                ET.SubElement(ci_rate, "timebase").text = str(fps)

                if src:
                    file_el = ET.SubElement(ci, "file", id=f"file-{file_idx}")
                    ET.SubElement(file_el, "name").text = os.path.basename(src)
                    ET.SubElement(file_el, "pathurl").text = f"file://localhost{src}"

                file_idx += 1

        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")

    def _frames_to_fcptime(proj: dict, fps: int) -> str:
        max_f = 0
        for t in proj.get("tracks", []):
            for c in t.get("clips", []):
                ef = c.get("endFrame", c.get("end_frame", 0))
                if ef > max_f:
                    max_f = ef
        return f"{max_f}/{fps}"
