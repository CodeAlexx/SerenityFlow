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

log = logging.getLogger(__name__)


def register_routes(app: FastAPI):
    """Register all REST routes on the FastAPI app."""

    def _state():
        from serenityflow.server.app import state
        return state

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

    # === SerenityFlow extensions ===

    @app.get("/api/sf/timeline/{prompt_id}")
    async def get_timeline(prompt_id: str):
        state = _state()
        if prompt_id in state.history:
            timeline = state.history[prompt_id].get("timeline", {})
            return JSONResponse(timeline)
        return JSONResponse({}, status_code=404)


__all__ = ["register_routes"]
