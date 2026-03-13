# SerenityFlow

A new inference engine for diffusion models. Graph-based workflow execution with integrated memory orchestration, block-level VRAM management, and ComfyUI custom node compatibility.

## What is this

SerenityFlow is a standalone inference engine that executes ComfyUI-format workflow JSONs through its own runtime. It is not a fork of ComfyUI, not a wrapper around diffusers, and not a monolithic pipeline. It is a new engine built from scratch with memory-aware graph execution at its core.

## How it's different

Every existing tool treats memory management and workflow execution as separate problems. SerenityFlow treats them as one coordinated system.

**ComfyUI** — Graph-based but the memory system is reactive. No prefetch, no budget, no patch tracking. The executor doesn't tell the memory system what's coming next. Nodes individually call `load_models_gpu` and hope there's room.

**Automatic1111 / Forge** — Monolithic pipeline scripts. No graph execution at all. Forge bolted on some memory optimizations (shared GPU memory, model patching) but it's still a linear pipeline with hardcoded stages. No composability.

**InvokeAI** — Proper architecture, graph executor, session management. Closest to what SerenityFlow is doing conceptually. But no block-level swap, no VRAM budgeting, no graph-aware prefetch. Their memory strategy is whole-model load/offload.

**Diffusers** — A library, not an engine. Provides pipelines but no graph execution, no memory orchestration, no workflow composition.

**SerenityFlow** — Stagehand block-swap with PatchLedger dirty tracking integrated into the H2D transfer path. Graph planner that computes model lifetimes and prefetch/evict points before execution starts. Residency-aware cache that moves tensors between GPU and CPU-pinned under VRAM pressure. Three-tier hook system where ControlNet blocks participate in the same VRAM budget as the base model. Execution timeline with per-node timing and VRAM snapshots. All through a workflow JSON interface.

Nobody else has the memory system talking to the graph planner talking to the patch tracker talking to the hook dispatcher. Those are all separate concerns in every other tool. In SerenityFlow they're one coordinated system.

## Architecture

```
serenityflow/
├── bridge/        — ComfyUI workflow parsing, model detection, Serenity API adapter
├── core/          — Type system, patch ledger, timeline, hooks, budget, conditioning
├── executor/      — Graph builder, topological runner, output cache, planner
├── memory/        — Stagehand coordinator, graph-aware prefetch/evict, patch integration
├── nodes/         — 275 node implementations (loaders, sampling, image, video, audio, model-specific)
├── compat/        — comfy.* shim package for custom node compatibility (50 modules)
└── cli.py         — Entry point
```

### Key subsystems

**Graph Planner** — Before execution begins, the planner walks the workflow graph and computes model lifetimes: when each model is first needed, when it's last used, and what can be evicted between those points. Prefetch decisions are made at graph compile time, not at execution time.

**Stagehand Integration** — Block-level GPU/CPU streaming for large models. Individual transformer blocks are transferred to GPU on demand, with a pinned memory pool and prefetch window. The coordinator detects model architecture automatically (Flux, SDXL, SD3, LTX-V, HunyuanVideo, WAN) and creates appropriate runtimes.

**PatchLedger** — Tracks which model blocks have been modified by LoRA/LoHA/LoKr/IA3 patches. During Stagehand's H2D transfer, only dirty blocks get patches reapplied from clean CPU source weights. Clean blocks transfer without extra computation.

**Residency Cache** — Model outputs cached with residency awareness. Under VRAM pressure, cached tensors migrate to CPU-pinned memory rather than being recomputed. Cache keys incorporate patch fingerprints so LoRA changes invalidate correctly.

**Hook System** — Three-tier hooks (model, block, attention) that participate in the VRAM budget. ControlNet conditioning blocks are scheduled through the same Stagehand runtime as the base model, not loaded separately.

### Compatibility layer

The `compat/` package provides a `comfy.*` namespace that custom nodes can import. It shims the top custom node import surface (model_management, model_patcher, utils, samplers, ops, ldm/attention, k_diffusion, folder_paths, nodes, server) and routes operations through SerenityFlow's infrastructure.

This is a translation surface, not a bypass. Everything routes through our memory coordinator and graph planner.

## Usage

```bash
# Run a ComfyUI workflow JSON
python -m serenityflow.cli --workflow workflow.json --model-dir /path/to/models

# With Stagehand block-swap for large models
python -m serenityflow.cli --workflow workflow.json --model-dir /path/to/models \
    --stagehand-pool-mb 8192 --stagehand-vram-budget 22000

# Verbose logging with telemetry
python -m serenityflow.cli --workflow workflow.json --model-dir /path/to/models \
    --stagehand-telemetry -v
```

## Tests

```bash
python -m pytest tests/ -q
# 604 passed
```

## Status

Active development. The engine runs workflows end-to-end. Custom node compatibility layer covers the import surface of the top 8 ComfyUI custom node repos (IP-Adapter Plus, ControlNet Aux, Impact Pack, AnimateDiff-Evolved, rgthree, KJNodes, WAS Suite, ComfyUI-GGUF).

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- [Stagehand](https://github.com/CodeAlexx/StageHand) (block-level GPU/CPU streaming)
- safetensors
- PyYAML

## License

MIT
