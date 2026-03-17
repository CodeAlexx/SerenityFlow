"""REST endpoints. ComfyUI-compatible paths and response formats."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid

import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger(__name__)

_CANVAS_WORKFLOWS_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "canvas", "workflows"))
_WORKFLOW_TEMPLATES_REPO_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "tests", "workflow_templates", "templates"))
_TEMPLATE_AUDIT_REPORT_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "tests", "template_audit_report.json"))


class _StripApiPrefixMiddleware(BaseHTTPMiddleware):
    """Strip /api/ prefix from request paths.

    ComfyUI frontend v1.34+ sends all API calls to /api/... (via fetchApi → apiURL).
    This middleware rewrites /api/foo → /foo so our routes work without duplication.
    Does NOT touch WebSocket upgrades (those go to /ws directly).
    """

    # Paths that should NOT be rewritten (static assets, frontend)
    _SKIP_PREFIXES = ("/assets/", "/scripts/", "/fonts/", "/cursor/", "/extensions/")

    async def dispatch(self, request, call_next):
        path = request.scope.get("path", "")
        if path.startswith("/api/") and not any(path.startswith(p) for p in self._SKIP_PREFIXES):
            # Rewrite /api/foo → /foo
            new_path = path[4:]  # strip "/api"
            request.scope["path"] = new_path
        return await call_next(request)


def register_routes(app: FastAPI):
    """Register all REST routes on the FastAPI app.

    ComfyUI frontend v1.34+ uses /api/ prefixed paths for all API calls
    (fetchApi → apiURL which prepends /api). We register routes at both
    /path and /api/path for compatibility.
    """

    def _state():
        from serenityflow.server.app import state
        return state

    def _load_template_runnability() -> dict[str, bool]:
        if not os.path.isfile(_TEMPLATE_AUDIT_REPORT_PATH):
            return {}
        try:
            with open(_TEMPLATE_AUDIT_REPORT_PATH, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception:
            return {}

        template_status = report.get("template_status")
        if not isinstance(template_status, dict):
            return {}

        result: dict[str, bool] = {}
        for rel_path, status in template_status.items():
            if not isinstance(rel_path, str) or not isinstance(status, dict):
                continue
            runnable = bool(status.get("runnable", True))
            if rel_path.startswith("templates/"):
                rel_path = rel_path[len("templates/"):]
            result[rel_path.replace("\\", "/")] = runnable
        return result

    template_runnability = _load_template_runnability()

    # Add /api/ prefix middleware BEFORE routes
    app.add_middleware(_StripApiPrefixMiddleware)

    def _get_all_registered():
        """Merge native registry + compat NODE_CLASS_MAPPINGS."""
        from serenityflow.nodes.registry import registry
        result = {}

        # Native nodes (NodeDef objects)
        for name, node_def in registry.list_all().items():
            result[name] = node_def

        # Compat nodes (class objects)
        try:
            from serenityflow.compat.nodes import NODE_CLASS_MAPPINGS
            for name, cls in NODE_CLASS_MAPPINGS.items():
                if name not in result:
                    result[name] = cls
        except ImportError:
            pass

        return result

    # === Prompt submission ===

    @app.post("/prompt")
    async def post_prompt(request: Request):
        from serenityflow.server.websocket import get_queue_info, send_event

        state = _state()
        data = await request.json()
        prompt = data.get("prompt", {})

        if not prompt:
            return JSONResponse({"error": "No prompt provided"}, status_code=400)

        # Validate nodes exist
        registered = _get_all_registered()
        errors = []
        output_nodes = []

        for node_id, node in prompt.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            if not class_type:
                continue

            # Check native registry (NodeDef)
            from serenityflow.nodes.registry import NodeDef
            found = class_type in registered
            if not found:
                errors.append(f"Node {node_id}: unknown class_type '{class_type}'")
                continue

            # Detect output nodes
            entry = registered[class_type]
            if isinstance(entry, NodeDef):
                if entry.is_output:
                    output_nodes.append(node_id)
            else:
                # Compat class
                if getattr(entry, "OUTPUT_NODE", False):
                    output_nodes.append(node_id)

        if errors:
            return JSONResponse({
                "error": "Validation failed",
                "node_errors": errors,
            }, status_code=400)

        if not output_nodes:
            output_nodes = list(prompt.keys())

        prompt_id = str(uuid.uuid4())
        item = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "extra_data": data.get("extra_data", {}),
            "output_nodes": output_nodes,
        }

        await state.prompt_queue.put(item)
        await send_event(state, "status", {"status": get_queue_info(state)})

        return JSONResponse({
            "prompt_id": prompt_id,
            "number": state.prompt_queue.qsize(),
        })

    # === Queue management ===

    @app.get("/queue")
    async def get_queue():
        state = _state()
        running = []
        if state.executing:
            running.append({"prompt_id": state.executing})
        return JSONResponse({
            "queue_running": running,
            "queue_pending": [],
        })

    @app.post("/queue")
    async def post_queue(request: Request):
        state = _state()
        data = await request.json()
        if data.get("clear"):
            while not state.prompt_queue.empty():
                try:
                    state.prompt_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        return JSONResponse({"status": "ok"})

    # === Interrupt ===

    @app.post("/interrupt")
    async def post_interrupt():
        state = _state()
        if state.runner:
            state.runner.interrupt()
        try:
            import comfy.model_management as mm
            mm.interrupt_current_processing(True)
        except Exception:
            pass
        return JSONResponse({"status": "ok"})

    # === History ===

    @app.get("/history")
    async def get_history(max_items: int = 200):
        state = _state()
        items = dict(list(state.history.items())[-max_items:])
        return JSONResponse(items)

    @app.get("/history/{prompt_id}")
    async def get_history_item(prompt_id: str):
        state = _state()
        if prompt_id in state.history:
            return JSONResponse({prompt_id: state.history[prompt_id]})
        return JSONResponse({}, status_code=404)

    @app.post("/history")
    async def post_history(request: Request):
        state = _state()
        data = await request.json()
        if data.get("clear"):
            state.history.clear()
        elif "delete" in data:
            for pid in data["delete"]:
                state.history.pop(pid, None)
        return JSONResponse({"status": "ok"})

    # === Free memory ===

    @app.post("/free")
    async def post_free(request: Request):
        data = await request.json()
        try:
            import comfy.model_management as mm
            if data.get("unload_models"):
                mm.unload_all_models()
            if data.get("free_memory"):
                mm.soft_empty_cache()
        except Exception:
            pass
        return JSONResponse({"status": "ok"})

    # === System stats ===

    @app.get("/system_stats")
    async def get_system_stats():
        stats = {
            "system": {
                "os": os.name,
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "N/A",
            },
            "devices": [],
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                stats["devices"].append({
                    "name": props.name,
                    "type": "cuda",
                    "index": i,
                    "vram_total": total,
                    "vram_free": free,
                    "torch_vram_total": total,
                    "torch_vram_free": free,
                })

        return JSONResponse(stats)

    # === Node info ===

    @app.get("/object_info")
    async def get_object_info():
        from serenityflow.nodes.registry import NodeDef

        registered = _get_all_registered()
        result = {}

        for name, entry in registered.items():
            if isinstance(entry, NodeDef):
                result[name] = {
                    "input": entry.input_types,
                    "output": list(entry.return_types),
                    "output_name": list(entry.return_names),
                    "name": name,
                    "display_name": entry.display_name,
                    "category": entry.category,
                    "output_node": entry.is_output,
                }
            else:
                # Compat class
                info = {
                    "input": {},
                    "output": [],
                    "output_name": [],
                    "name": name,
                    "display_name": name,
                    "category": getattr(entry, "CATEGORY", ""),
                    "output_node": getattr(entry, "OUTPUT_NODE", False),
                }
                try:
                    if hasattr(entry, "INPUT_TYPES") and callable(entry.INPUT_TYPES):
                        info["input"] = entry.INPUT_TYPES()
                except Exception:
                    pass
                try:
                    info["output"] = list(getattr(entry, "RETURN_TYPES", ()))
                    info["output_name"] = list(
                        getattr(entry, "RETURN_NAMES", info["output"])
                    )
                except Exception:
                    pass
                result[name] = info

        return JSONResponse(result)

    @app.get("/object_info/{node_class}")
    async def get_object_info_single(node_class: str):
        from serenityflow.nodes.registry import NodeDef

        registered = _get_all_registered()
        if node_class not in registered:
            return JSONResponse({}, status_code=404)

        # Reuse logic from get_object_info for single node
        entry = registered[node_class]
        if isinstance(entry, NodeDef):
            info = {
                "input": entry.input_types,
                "output": list(entry.return_types),
                "output_name": list(entry.return_names),
                "name": node_class,
                "display_name": entry.display_name,
                "category": entry.category,
                "output_node": entry.is_output,
            }
        else:
            info = {
                "input": {},
                "output": [],
                "output_name": [],
                "name": node_class,
                "display_name": node_class,
                "category": getattr(entry, "CATEGORY", ""),
                "output_node": getattr(entry, "OUTPUT_NODE", False),
            }
            try:
                if hasattr(entry, "INPUT_TYPES") and callable(entry.INPUT_TYPES):
                    info["input"] = entry.INPUT_TYPES()
            except Exception:
                pass
            try:
                info["output"] = list(getattr(entry, "RETURN_TYPES", ()))
                info["output_name"] = list(
                    getattr(entry, "RETURN_NAMES", info["output"])
                )
            except Exception:
                pass

        return JSONResponse({node_class: info})

    # === Models ===

    @app.get("/models")
    async def get_models():
        try:
            import folder_paths
            folders = list(folder_paths.folder_names_and_paths.keys())
            return JSONResponse(folders)
        except Exception:
            return JSONResponse([])

    @app.get("/models/{folder}")
    async def get_models_folder(folder: str):
        try:
            import folder_paths
            files = folder_paths.get_filename_list(folder)
            return JSONResponse(files)
        except Exception:
            return JSONResponse([], status_code=404)

    @app.get("/embeddings")
    async def get_embeddings():
        try:
            import folder_paths
            return JSONResponse(folder_paths.get_filename_list("embeddings"))
        except Exception:
            return JSONResponse([])

    # === File serving ===

    @app.get("/view")
    async def view_file(
        filename: str,
        subfolder: str = "",
        type: str = "output",
        preview: str | None = None,
        channel: str | None = None,
        quality: int = 90,
    ):
        state = _state()

        if type == "output":
            base = state.output_dir
        elif type == "input":
            base = state.input_dir
        elif type == "temp":
            base = state.temp_dir
        else:
            return Response(status_code=400)

        filepath = os.path.join(base, subfolder, filename)
        filepath = os.path.realpath(filepath)

        # Security: prevent path traversal
        base_real = os.path.realpath(base)
        if not filepath.startswith(base_real + os.sep) and filepath != base_real:
            return Response(status_code=403)

        if not os.path.isfile(filepath):
            return Response(status_code=404)

        return FileResponse(filepath)

    @app.get("/view_metadata/{folder_name}")
    async def view_metadata(folder_name: str, filename: str = ""):
        try:
            import folder_paths
            path = folder_paths.get_full_path(folder_name, filename)
        except Exception:
            return JSONResponse({}, status_code=404)

        if path is None or not os.path.exists(path):
            return JSONResponse({}, status_code=404)

        if path.endswith(".safetensors"):
            try:
                import comfy.utils
                header = comfy.utils.safetensors_header(path)
                metadata = header.get("__metadata__", {}) if header else {}
                return JSONResponse(metadata)
            except Exception:
                return JSONResponse({})
        return JSONResponse({})

    # === Upload ===

    @app.post("/upload/image")
    async def upload_image(
        image: UploadFile = File(...),
        overwrite: str = Form("false"),
        type: str = Form("input"),
        subfolder: str = Form(""),
    ):
        state = _state()

        if type == "input":
            base = state.input_dir
        elif type == "temp":
            base = state.temp_dir
        else:
            return JSONResponse({"error": "Invalid type"}, status_code=400)

        upload_dir = os.path.join(base, subfolder)
        os.makedirs(upload_dir, exist_ok=True)

        filename = image.filename or "upload.png"

        # Sanitize filename
        filename = os.path.basename(filename)
        filepath = os.path.join(upload_dir, filename)

        # Security: verify resolved path is within base
        filepath_real = os.path.realpath(filepath)
        base_real = os.path.realpath(base)
        if not filepath_real.startswith(base_real + os.sep) and filepath_real != base_real:
            return JSONResponse({"error": "Invalid path"}, status_code=403)

        # Don't overwrite unless asked
        if overwrite != "true" and os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                filename = f"{name}_{counter}{ext}"
                filepath = os.path.join(upload_dir, filename)
                counter += 1

        # Limit upload size (100MB)
        content = await image.read()
        if len(content) > 100 * 1024 * 1024:
            return JSONResponse({"error": "File too large"}, status_code=413)

        with open(filepath, "wb") as f:
            f.write(content)

        return JSONResponse({
            "name": filename,
            "subfolder": subfolder,
            "type": type,
        })

    @app.post("/upload/mask")
    async def upload_mask(
        image: UploadFile = File(...),
        overwrite: str = Form("false"),
        original_ref: str = Form("{}"),
    ):
        return await upload_image(image, overwrite, "input", "")

    # === Features ===

    @app.get("/features")
    async def get_features():
        return JSONResponse([])

    # === ComfyUI frontend compatibility endpoints ===

    @app.get("/settings")
    async def get_settings():
        return JSONResponse({})

    @app.get("/settings/{setting_id}")
    async def get_setting(setting_id: str):
        return JSONResponse(None)

    @app.post("/settings")
    async def post_settings(request: Request):
        return JSONResponse({"status": "ok"})

    @app.post("/settings/{setting_id}")
    async def post_setting(setting_id: str, request: Request):
        return JSONResponse({"status": "ok"})

    @app.get("/users")
    async def get_users():
        return JSONResponse({"storage": "server", "users": {"default": "default"}})

    @app.get("/userdata")
    async def get_userdata():
        return JSONResponse([])

    @app.get("/userdata/{path:path}")
    async def get_userdata_file(path: str):
        # Special case: user.css returns empty
        if path.endswith(".css"):
            return Response("", media_type="text/css")
        if path.endswith(".json"):
            return JSONResponse({})
        return Response("", status_code=200)

    @app.post("/userdata/{path:path}")
    async def post_userdata_file(path: str, request: Request):
        return JSONResponse({"status": "ok"})

    @app.get("/api/userdata/{path:path}")
    async def get_api_userdata_file(path: str):
        if path.endswith(".css"):
            return Response("", media_type="text/css")
        if path.endswith(".json"):
            return JSONResponse({})
        return Response("", status_code=200)

    @app.get("/extensions")
    async def get_extensions():
        return JSONResponse([])

    @app.get("/folder_paths")
    async def get_folder_paths():
        import folder_paths as fp
        result = {}
        for cat, (paths, exts) in fp.folder_names_and_paths.items():
            result[cat] = {"paths": paths, "extensions": list(exts)}
        return JSONResponse(result)

    @app.post("/folder_paths/add")
    async def add_folder_path(request: Request):
        """Add an extra model search directory at runtime."""
        data = await request.json()
        path = data.get("path", "")
        if not path or not os.path.isdir(os.path.expanduser(path)):
            return JSONResponse({"error": f"Directory not found: {path}"}, status_code=400)
        import folder_paths as fp
        fp.add_extra_model_dirs([os.path.expanduser(path)])
        return JSONResponse({"status": "ok", "path": path})

    @app.get("/nodes")
    async def get_nodes():
        return JSONResponse([])

    @app.get("/nodes/search")
    async def search_nodes():
        return JSONResponse([])

    @app.get("/bulk/nodes/versions")
    async def get_bulk_node_versions():
        return JSONResponse({})

    @app.get("/logs")
    async def get_logs():
        return JSONResponse([])

    @app.get("/logs/raw")
    async def get_logs_raw():
        return Response("", media_type="text/plain")

    @app.get("/logs/subscribe")
    async def logs_subscribe():
        return JSONResponse([])

    @app.get("/internal/logs")
    async def get_internal_logs():
        return JSONResponse({"entries": []})

    @app.get("/workflow_templates")
    async def get_workflow_templates():
        return JSONResponse([])

    @app.get("/global_subgraphs")
    async def get_global_subgraphs():
        return JSONResponse([])

    @app.get("/global_subgraphs/{name}")
    async def get_global_subgraph(name: str):
        return JSONResponse({}, status_code=404)

    @app.get("/user.css")
    async def get_user_css():
        return Response("", media_type="text/css")

    # === Templates ===

    @app.get("/templates")
    async def list_templates():
        result = []
        seen_urls = set()

        def _collect(dir_path: str, url_prefix: str) -> None:
            if not os.path.isdir(dir_path):
                return
            for root, dirs, files in os.walk(dir_path):
                dirs[:] = sorted(d for d in dirs if d and not d.startswith("."))
                files.sort()
                for filename in files:
                    if filename.startswith(".") or not filename.lower().endswith(".json"):
                        continue
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, dir_path)
                    rel_url = rel_path.replace(os.sep, "/")
                    if url_prefix == "workflow_templates" and template_runnability.get(rel_url) is False:
                        continue
                    url = f"{url_prefix}/{rel_url}"
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    base_name = os.path.splitext(rel_url)[0]
                    display_name = base_name.replace("/", " ").replace("_", " ")
                    display_name = " ".join(display_name.split())
                    if display_name:
                        display_name = display_name.title()
                    else:
                        display_name = os.path.splitext(filename)[0]
                    result.append({
                        "name": display_name,
                        "file": rel_url,
                        "url": url,
                    })

        _collect(_CANVAS_WORKFLOWS_DIR, "workflows")
        _collect(_WORKFLOW_TEMPLATES_REPO_DIR, "workflow_templates")
        return JSONResponse(result)

    # === SerenityFlow extensions ===

    @app.get("/sf/timeline/{prompt_id}")
    async def get_timeline(prompt_id: str):
        state = _state()
        if prompt_id in state.history:
            timeline = state.history[prompt_id].get("timeline", {})
            return JSONResponse(timeline)
        return JSONResponse({}, status_code=404)

    # === Prompt enhancement ===

    @app.post("/enhance_prompt")
    async def enhance_prompt(request: Request):
        """Enhance a generation prompt. Falls back to rule-based when no LLM available."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        prompt = body.get("prompt", "").strip()
        arch = body.get("arch", "flux")

        if not prompt:
            return JSONResponse({"enhanced": prompt, "source": "rules"})

        # Rule-based enhancement
        enhanced = _enhance_rule_based(prompt, arch)
        return JSONResponse({"enhanced": enhanced, "source": "rules"})

    # === Output file management ===

    @app.get("/output_files")
    async def list_output_files():
        """List recently generated output files."""
        state = _state()
        output_dir = state.output_dir
        if not os.path.isdir(output_dir):
            return JSONResponse([])
        files = []
        try:
            entries = sorted(
                os.listdir(output_dir),
                key=lambda x: os.path.getmtime(os.path.join(output_dir, x)),
                reverse=True,
            )
            for name in entries[:50]:
                path = os.path.join(output_dir, name)
                if not os.path.isfile(path):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext not in (".png", ".jpg", ".jpeg", ".webp", ".mp4"):
                    continue
                stat = os.stat(path)
                files.append({
                    "name": name,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                    "type": "video" if ext == ".mp4" else "image",
                })
        except Exception:
            pass
        return JSONResponse(files)

    @app.delete("/output_files/{filename}")
    async def delete_output_file(filename: str):
        """Delete a generated output file."""
        state = _state()
        output_dir = state.output_dir
        # Security: sanitize filename
        safe_name = os.path.basename(filename)
        path = os.path.join(output_dir, safe_name)
        if not os.path.isfile(path):
            return JSONResponse({"error": "Not found"}, status_code=404)
        # Verify path is within output_dir
        if not os.path.realpath(path).startswith(os.path.realpath(output_dir)):
            return JSONResponse({"error": "Invalid path"}, status_code=403)
        try:
            os.unlink(path)
            return JSONResponse({"status": "deleted"})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/open_output_dir")
    async def open_output_dir():
        """Open output directory in OS file manager."""
        import platform
        import subprocess

        state = _state()
        output_dir = state.output_dir
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", output_dir])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", output_dir])
            else:
                subprocess.Popen(["xdg-open", output_dir])
            return JSONResponse({"status": "ok"})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)})

    # === Stagehand settings ===

    @app.post("/stagehand_settings")
    async def apply_stagehand_settings(request: Request):
        """Apply Stagehand memory settings at runtime."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        # Stub — Stagehand runtime config not yet implemented
        return JSONResponse({
            "status": "ok",
            "message": "Settings received. Runtime Stagehand config will be supported in a future update.",
        })

    # === Frontend static files ===
    # Priority: SerenityFlow canvas > ComfyUI frontend > API-only

    _canvas_dir = os.path.join(os.path.dirname(__file__), "..", "canvas")
    _canvas_dir = os.path.realpath(_canvas_dir)
    _use_canvas = os.path.isfile(os.path.join(_canvas_dir, "index.html"))

    if _use_canvas:
        from fastapi.staticfiles import StaticFiles

        @app.get("/")
        async def serve_index():
            return FileResponse(os.path.join(_canvas_dir, "index.html"))

        # Mount canvas static subdirectories
        for subdir in ("css", "js", "assets", "workflows"):
            subpath = os.path.join(_canvas_dir, subdir)
            if os.path.isdir(subpath):
                app.mount(
                    f"/{subdir}",
                    StaticFiles(directory=subpath),
                    name=f"canvas_{subdir}",
                )

        if os.path.isdir(_WORKFLOW_TEMPLATES_REPO_DIR):
            app.mount(
                "/workflow_templates",
                StaticFiles(directory=_WORKFLOW_TEMPLATES_REPO_DIR),
                name="workflow_templates",
            )

        # Serve loose files from canvas root
        @app.get("/{filename:path}")
        async def serve_canvas_file(filename: str):
            filepath = os.path.join(_canvas_dir, filename)
            filepath = os.path.realpath(filepath)
            base_real = os.path.realpath(_canvas_dir)
            if not filepath.startswith(base_real + os.sep) and filepath != base_real:
                return Response(status_code=403)
            if os.path.isfile(filepath):
                return FileResponse(filepath)
            return Response(status_code=404)

        log.info("SerenityFlow canvas served from: %s", _canvas_dir)

    else:
        _frontend_dir = _find_frontend_dir()
        if _frontend_dir:
            from fastapi.staticfiles import StaticFiles

            @app.get("/")
            async def serve_index():
                index = os.path.join(_frontend_dir, "index.html")
                if os.path.isfile(index):
                    return FileResponse(index)
                return Response("SerenityFlow v2 server running.", status_code=200)

            for subdir in ("assets", "scripts", "fonts", "cursor", "extensions"):
                subpath = os.path.join(_frontend_dir, subdir)
                if os.path.isdir(subpath):
                    app.mount(f"/{subdir}", StaticFiles(directory=subpath), name=f"static_{subdir}")

            @app.get("/{filename:path}")
            async def serve_frontend_file(filename: str):
                filepath = os.path.join(_frontend_dir, filename)
                filepath = os.path.realpath(filepath)
                base_real = os.path.realpath(_frontend_dir)
                if not filepath.startswith(base_real + os.sep) and filepath != base_real:
                    return Response(status_code=403)
                if os.path.isfile(filepath):
                    return FileResponse(filepath)
                index = os.path.join(_frontend_dir, "index.html")
                if os.path.isfile(index):
                    return FileResponse(index)
                return Response(status_code=404)

            log.info("ComfyUI frontend served from: %s", _frontend_dir)
        else:
            @app.get("/")
            async def serve_root():
                return Response("SerenityFlow v2 server running. No frontend found.", status_code=200)

            log.warning("No frontend found. Server running in API-only mode.")


def _enhance_rule_based(prompt: str, arch: str) -> str:
    """Simple rule-based prompt enhancement."""
    lower = prompt.lower()
    parts = []

    if "light" not in lower and "lit" not in lower:
        parts.append("natural lighting")
    if "detail" not in lower and "quality" not in lower:
        parts.append("highly detailed")
    if "composit" not in lower and "angle" not in lower:
        parts.append("professional composition")

    arch_suffix = {
        "flux": "masterful execution, 8k",
        "sdxl": "masterpiece, best quality, sharp focus",
        "sd3": "masterpiece, best quality",
        "sd15": "best quality, sharp focus",
        "ltxv": "smooth motion, cinematic quality",
        "wan": "smooth motion, high quality",
    }
    parts.append(arch_suffix.get(arch, "high quality"))

    return prompt.rstrip(".,") + ", " + ", ".join(parts)


def _find_frontend_dir() -> str | None:
    """Locate a ComfyUI frontend installation."""
    # Check env var first
    env_dir = os.environ.get("SERENITYFLOW_FRONTEND_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # Check common locations
    candidates = [
        # Local frontend in project
        os.path.join(os.path.dirname(__file__), "..", "..", "web"),
        # SwarmUI's ComfyUI frontend (use latest version)
        os.path.expanduser("~/SwarmUI/dlbackend/ComfyUI/web_custom_versions/Comfy-Org_ComfyUI_frontend"),
        # Direct ComfyUI install
        os.path.expanduser("~/ComfyUI/web"),
    ]

    for candidate in candidates:
        candidate = os.path.realpath(candidate)
        if not os.path.isdir(candidate):
            continue

        # If it's a versioned directory, pick the latest
        if os.path.isfile(os.path.join(candidate, "index.html")):
            return candidate

        # Check for version subdirectories
        try:
            versions = sorted(
                [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))],
                key=lambda v: [int(x) for x in v.split(".") if x.isdigit()],
            )
            if versions:
                latest = os.path.join(candidate, versions[-1])
                if os.path.isfile(os.path.join(latest, "index.html")):
                    return latest
        except (ValueError, OSError):
            continue

    return None


__all__ = ["register_routes"]
