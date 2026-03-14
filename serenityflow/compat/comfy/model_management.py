"""Compatibility shim for comfy.model_management.

Custom nodes import this for device queries and model loading.
Routes through Stagehand for memory operations.
"""
from __future__ import annotations

import platform
from enum import Enum

import torch

try:
    import psutil
except ImportError:
    psutil = None


class VRAMState(Enum):
    DISABLED = 0
    NO_VRAM = 1
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


vram_state = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

# === Device queries ===


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    return torch.device("cpu")


def intermediate_device():
    return torch.device("cpu")


def unet_offload_device():
    return torch.device("cpu")


def text_encoder_device():
    return get_torch_device()


def text_encoder_offload_device():
    return torch.device("cpu")


def vae_device():
    return get_torch_device()


def vae_offload_device():
    return torch.device("cpu")


def is_device_cpu(device):
    if hasattr(device, "type"):
        return device.type == "cpu"
    return False


def is_device_cuda(device):
    if hasattr(device, "type"):
        return device.type == "cuda"
    return False


def is_device_mps(device):
    if hasattr(device, "type"):
        return device.type == "mps"
    return False


# === Hardware detection ===


def is_nvidia():
    return torch.cuda.is_available() and torch.version.cuda is not None


def is_amd():
    return torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip is not None


# === Memory queries ===


def get_free_memory(dev=None, torch_free_too=False):
    if dev is None:
        dev = get_torch_device()
    if is_device_cpu(dev):
        if psutil is not None:
            free = psutil.virtual_memory().available
        else:
            free = 8 * 1024 * 1024 * 1024  # fallback 8GB
        return (free, free) if torch_free_too else free
    stats = torch.cuda.memory_stats(dev)
    mem_active = stats["active_bytes.all.current"]
    mem_reserved = stats["reserved_bytes.all.current"]
    mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
    mem_free_torch = mem_reserved - mem_active
    total = mem_free_cuda + mem_free_torch
    return (total, mem_free_torch) if torch_free_too else total


def get_total_memory(dev=None, torch_total_too=False):
    if dev is None:
        dev = get_torch_device()
    if is_device_cpu(dev):
        if psutil is not None:
            total = psutil.virtual_memory().total
        else:
            total = 32 * 1024 * 1024 * 1024
        return (total, total) if torch_total_too else total
    _, total = torch.cuda.mem_get_info(dev)
    if torch_total_too:
        stats = torch.cuda.memory_stats(dev)
        return (total, stats["reserved_bytes.all.current"])
    return total


# === Model loading → Stagehand ===

current_loaded_models = []


def load_models_gpu(models, memory_required=0, force_patch_weights=False,
                    minimum_memory_required=None, force_full_load=False):
    for m in models:
        if hasattr(m, "model") and isinstance(m.model, torch.nn.Module):
            # Move model to GPU if it's on CPU (small models only;
            # large models use block-level offloading hooks instead)
            try:
                device = get_torch_device()
                m.model.to(device)
            except Exception:
                pass


def load_model_gpu(model):
    return load_models_gpu([model])


def loaded_models(only_currently_used=False):
    return []


# === Dtype helpers ===


def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device and is_device_cpu(device):
        return False
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device or get_torch_device())
        return props.major >= 7
    return False


def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device and is_device_cpu(device):
        return False
    return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False


def unet_dtype(device=None, model_params=0, supported_dtypes=None, weight_dtype=None):
    if weight_dtype is not None:
        return weight_dtype
    if should_use_bf16(device):
        return torch.bfloat16
    if should_use_fp16(device):
        return torch.float16
    return torch.float32


def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=None):
    if weight_dtype in (torch.float32, torch.float64):
        return None
    if should_use_bf16(inference_device) and weight_dtype == torch.bfloat16:
        return None
    if should_use_fp16(inference_device) and weight_dtype == torch.float16:
        return None
    if should_use_bf16(inference_device):
        return torch.bfloat16
    return torch.float32


def text_encoder_dtype(device=None):
    return torch.float16


def vae_dtype(device=None, allowed_dtypes=None):
    if allowed_dtypes:
        for d in allowed_dtypes:
            if d == torch.bfloat16 and should_use_bf16(device):
                return d
    return torch.float32


def dtype_size(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def maximum_vram_for_weights(device=None):
    return get_total_memory(device) * 0.88 - minimum_inference_memory()


def minimum_inference_memory():
    return int(1024 * 1024 * 1024 * 0.8) + extra_reserved_memory()


def extra_reserved_memory():
    return 400 * 1024 * 1024


# === Cache management ===


def soft_empty_cache(force=False):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def unload_all_models():
    soft_empty_cache()


def cleanup_models():
    pass


def cleanup_models_gc():
    pass


# === Attention ===

XFORMERS_IS_AVAILABLE = False
ENABLE_PYTORCH_ATTENTION = True


def xformers_enabled():
    return XFORMERS_IS_AVAILABLE


def pytorch_attention_enabled():
    return ENABLE_PYTORCH_ATTENTION


# === Interrupt ===

interrupt_processing = False


class InterruptProcessingException(Exception):
    pass


def interrupt_current_processing(value=True):
    global interrupt_processing
    interrupt_processing = value


def processing_interrupted():
    return interrupt_processing


def throw_exception_if_processing_interrupted():
    if interrupt_processing:
        raise InterruptProcessingException()


# === Misc ===


def module_size(module):
    total = 0
    sd = module.state_dict()
    for k in sd:
        total += sd[k].nbytes
    return total


def get_torch_device_name(device):
    if hasattr(device, "type"):
        if device.type == "cuda":
            try:
                return f"{device} {torch.cuda.get_device_name(device)}"
            except Exception:
                return str(device)
        return str(device.type)
    return str(device)


def supports_dtype(device, dtype):
    return True


def supports_cast(device, dtype):
    return True


def device_supports_non_blocking(device):
    return not is_device_cpu(device)


def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, **kwargs):
    if device is None and dtype is None and not copy:
        return weight
    return weight.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)


def get_autocast_device(dev):
    if hasattr(dev, "type"):
        return dev.type
    return "cuda"


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


WINDOWS = platform.system() == "Windows"

NUM_STREAMS = 0


def get_offload_stream(device):
    return None


def sync_stream(device, stream):
    pass
