"""Every import pattern from the top 8 custom nodes must resolve."""
from __future__ import annotations

import os
import sys

# Wire compat path the same way cli.py does
compat_dir = os.path.join(os.path.dirname(__file__), "..", "serenityflow", "compat")
compat_dir = os.path.abspath(compat_dir)
if compat_dir not in sys.path:
    sys.path.insert(0, compat_dir)


def test_core_imports():
    import comfy
    import comfy.model_management
    import comfy.utils
    import comfy.model_patcher
    import comfy.sd
    import comfy.samplers
    import comfy.ops
    import comfy.hooks
    import comfy.lora
    import comfy.sample
    import comfy.conds
    import comfy.float
    import comfy.latent_formats
    import comfy.model_sampling
    import comfy.model_base
    import comfy.model_detection
    import comfy.patcher_extension
    import comfy.controlnet
    import comfy.clip_vision
    import comfy.diffusers_convert
    import comfy.cli_args


def test_ldm_imports():
    import comfy.ldm.modules.attention
    import comfy.ldm.modules.diffusionmodules.openaimodel
    import comfy.ldm.modules.diffusionmodules.util
    import comfy.ldm.flux.math
    import comfy.ldm.lightricks.model
    import comfy.ldm.common_dit
    from comfy.ldm.modules.attention import optimized_attention, CrossAttention, FeedForward
    from comfy.ldm.flux.math import apply_rope
    from comfy.ldm.modules.diffusionmodules.util import timestep_embedding, zero_module
    from comfy.ldm.lightricks.model import apply_rotary_emb


def test_k_diffusion_imports():
    import comfy.k_diffusion.sampling
    import comfy.k_diffusion.sa_solver
    from comfy.k_diffusion.sampling import to_d, sample_euler, sample_dpmpp_2m
    from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler


def test_external_imports():
    import folder_paths
    import nodes
    import server
    import execution
    import latent_preview
    from nodes import MAX_RESOLUTION
    assert MAX_RESOLUTION == 16384


def test_comfy_api_imports():
    from comfy_api.latest import io
    assert hasattr(io, "IMAGE")
    assert hasattr(io, "MODEL")


def test_comfy_execution_imports():
    from comfy_execution import graph_utils
    from comfy_execution import validation


def test_comfy_types_imports():
    from comfy.comfy_types import IO
    from comfy.comfy_types.node_typing import IO as IO2, ComfyNodeABC
    assert IO is IO2


def test_model_management_functions():
    import comfy.model_management as mm
    dev = mm.get_torch_device()
    assert dev is not None
    free = mm.get_free_memory()
    assert isinstance(free, int)
    mm.soft_empty_cache()
    assert not mm.processing_interrupted()
    assert mm.intermediate_device().type == "cpu"
    assert mm.unet_offload_device().type == "cpu"
    assert isinstance(mm.VRAMState.NORMAL_VRAM, mm.VRAMState)


def test_model_patcher_interface():
    from comfy.model_patcher import ModelPatcher
    mp = ModelPatcher(model=None, load_device=None, offload_device=None)
    clone = mp.clone()
    assert isinstance(clone, ModelPatcher)
    mp.set_model_attn1_patch(lambda x: x)
    mp.set_model_attn2_replace(lambda x: x, "block", 0)
    mp.set_model_sampler_cfg_function(lambda args: args["cond"])
    assert "attn1" in mp.model_options["transformer_options"]["patches"]
    assert "attn2" in mp.model_options["transformer_options"]["patches_replace"]


def test_folder_paths_functions():
    import folder_paths
    paths = folder_paths.get_folder_paths("checkpoints")
    assert isinstance(paths, list)
    assert len(paths) > 0
    folder_paths.add_model_folder_path("custom_cat", "/tmp/test_models")
    assert "/tmp/test_models" in folder_paths.get_folder_paths("custom_cat")


def test_module_identity():
    """Same module object regardless of import style."""
    import comfy.model_management
    import comfy.model_management as mm
    from comfy import model_management
    assert comfy.model_management is mm
    assert comfy.model_management is model_management


def test_prompt_server_singleton():
    import server
    assert server.PromptServer.instance is not None
    assert hasattr(server.PromptServer.instance, "routes")
    assert hasattr(server.PromptServer.instance, "send_sync")
    assert hasattr(server.PromptServer.instance, "client_id")
    assert hasattr(server.PromptServer.instance, "last_node_id")


def test_nodes_class_mappings():
    import nodes
    assert isinstance(nodes.NODE_CLASS_MAPPINGS, dict)
    assert len(nodes.NODE_CLASS_MAPPINGS) > 0
    assert "KSampler" in nodes.NODE_CLASS_MAPPINGS
    assert "SaveImage" in nodes.NODE_CLASS_MAPPINGS
    assert "CLIPTextEncode" in nodes.NODE_CLASS_MAPPINGS
    assert "LoraLoader" in nodes.NODE_CLASS_MAPPINGS


def test_utils_progress_bar():
    from comfy.utils import ProgressBar
    pbar = ProgressBar(100)
    pbar.update(10)
    assert pbar.current == 10
    pbar.update_absolute(50)
    assert pbar.current == 50


def test_utils_state_dict_helpers():
    import torch
    from comfy.utils import state_dict_prefix_replace, calculate_parameters

    sd = {"model.layer1.weight": torch.zeros(10, 10), "model.layer2.bias": torch.zeros(10)}
    replaced = state_dict_prefix_replace(sd, {"model.": "new."})
    assert "new.layer1.weight" in replaced
    assert "new.layer2.bias" in replaced

    params = calculate_parameters(sd, "model.")
    assert params == 110


def test_sampler_names():
    from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES
    assert "euler" in SAMPLER_NAMES
    assert "karras" in SCHEDULER_NAMES
    assert "dpmpp_2m" in SAMPLER_NAMES
    assert "normal" in SCHEDULER_NAMES


def test_model_base_types():
    from comfy.model_base import ModelType, BaseModel, SD15, SDXL, Flux
    assert ModelType.EPS.value == 1
    assert ModelType.FLUX.value == 8
    m = SD15()
    assert isinstance(m, BaseModel)
    x = SDXL()
    assert x.adm_channels == 2816


def test_ops_classes():
    from comfy.ops import disable_weight_init, manual_cast
    import torch
    lin = disable_weight_init.Linear(10, 20)
    assert isinstance(lin, torch.nn.Linear)
    mc_lin = manual_cast.Linear(10, 20)
    assert mc_lin.comfy_cast_weights is True


def test_hooks_classes():
    from comfy.hooks import HookGroup, HookKeyframeGroup, HookKeyframe, EnumHookMode
    hg = HookGroup()
    assert len(hg) == 0
    kfg = HookKeyframeGroup()
    kfg.add(HookKeyframe(strength=0.5))
    assert len(kfg.keyframes) == 1


def test_controlnet_base():
    from comfy.controlnet import ControlBase, broadcast_image_to
    import torch
    cb = ControlBase()
    cb.set_cond_hint(torch.zeros(1, 3, 64, 64), strength=0.8)
    assert cb.strength == 0.8

    t = torch.ones(1, 3, 64, 64)
    result = broadcast_image_to(t, 4, 1)
    assert result.shape[0] == 4


def test_patcher_extension():
    from comfy.patcher_extension import CallbacksMP, WrappersMP, WrapperExecutor
    assert CallbacksMP.ON_CLONE == "on_clone"
    assert WrappersMP.DIFFUSION_MODEL == "diffusion_model"


def test_latent_formats():
    from comfy.latent_formats import SD15, SDXL, Flux, SD3
    assert SD15.scale_factor == 0.18215
    assert SDXL.scale_factor == 0.13025
    f = Flux()
    processed = f.process_in(1.0)
    assert processed != 1.0


def test_cli_args():
    from comfy.cli_args import args
    assert hasattr(args, "listen")
    assert hasattr(args, "port")
    assert args.port == 8188


def test_model_sampling_classes():
    from comfy.model_sampling import ModelSamplingDiscrete, ModelSamplingFlux
    msd = ModelSamplingDiscrete()
    assert msd.sigma_min is not None
    assert msd.sigma_max is not None
    msf = ModelSamplingFlux()
    assert float(msf.sigma_max) == 1.0


def test_sd_load_functions():
    from comfy.sd import load_checkpoint_guess_config, load_lora_for_models, CLIP, VAE
    model, clip, vae, cv = load_checkpoint_guess_config("/fake/path")
    assert model is not None
    assert clip is not None
    assert vae is not None


def test_server_route_registration():
    """Two custom nodes registering routes should not collide."""
    import server
    ps = server.PromptServer.instance
    initial = len(ps.routes)

    @ps.routes.get("/api/custom1")
    def handler1():
        pass

    @ps.routes.post("/api/custom2")
    def handler2():
        pass

    assert len(ps.routes) == initial + 2
