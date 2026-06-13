# FP8 LTX-2.3 Session Handoff — 2026-03-18

## What Was Done This Session

### FP8 Gemma Text Encoder Loading — FIXED

The FP8 Gemma 12B text encoder (`gemma-3-12b-it-fp8`, 13GB) now loads and runs correctly in SerenityFlow v2. Three bugs were found and fixed in `serenityflow/bridge/serenity_api.py`:

**Bug 1: Key format mismatch (weights silently not loaded)**
- FP8 Gemma file uses HuggingFace key format: `model.layers.0.*`
- Standard bf16 Gemma for LTX uses: `language_model.model.layers.0.*`
- The SD ops only matched the standard format → all 626 language model keys from the FP8 file were silently dropped
- **Fix**: Created `_FP8GemmaKeyRemapLoader` class (~line 3890) that wraps the `StateDictLoader` and remaps `model.*` → `language_model.model.*` BEFORE SD ops processing

**Bug 2: Meta tensor crash on `.to(device)` (vision tower + lm_head)**
- ltx_core hardcodes a 27-layer SigLIP vision tower config, but FP8 file only has 10 layers → 17 layers left on meta device
- `lm_head.weight` is tied to `embed_tokens.weight` — tie breaks after `load_state_dict(assign=True)`, leaving lm_head on meta
- PyTorch `named_parameters()` deduplicates by `data_ptr()` — ALL meta tensors share `data_ptr()==0`, so most get skipped
- **Fix**: `_materialise_meta_params()` (~line 2801) iterates `named_modules()` and checks `_parameters`/`_buffers` dicts directly. Patched into `SingleGPUModelBuilder._return_model` via `_safe_return_model`

**Bug 3: Dtype mismatch in feature extractor**
- CPU language model outputs float32 hidden states, but feature extractor weights are bf16
- **Fix**: Cast hidden states to feature extractor dtype before calling `self.feature_extractor()` in `patched_precompute()`

**Config change**: Replaced symlinked `config.json` in `~/.serenity/models/text_encoders/gemma-3-12b-it-fp8/` with a standalone copy (vision_config removed, though ltx_core ignores it anyway since it uses hardcoded GEMMA3_CONFIG_FOR_LTX)

### Video Generation — WORKS BUT HAS GRID ARTIFACTS

Generated a 97-frame video at 512x320 using:
- FP8 checkpoint: `ltx-2.3-22b-dev-fp8.safetensors`
- FP8 Gemma text encoder (the fix above)
- Distilled mode, 8 steps + 3 refinement steps
- Stagehand block-swapping (transformer too large for 24GB VRAM after dequant)

The text encoder works correctly (prompt is understood, golden retriever + meadow are recognizable). But the video has **heavy grid/crosshatch artifacts** — this is a KNOWN pre-existing bug in the transformer FP8 dequant path, NOT related to the Gemma fix.

## The Grid Artifact Bug — UNSOLVED

There is a detailed handoff doc at `/home/alex/serenity/output/HANDOFF_FP8_GRID_ARTIFACTS.md` (written by a previous session). Key points:

**Root cause identified**: `_move_non_blocks_to_device()` in `serenity_api.py` casts ALL non-block params to bf16 unconditionally — including FP8 params like `proj_in` and `proj_out`. This cast does `fp8_value.to(bf16)` WITHOUT multiplying by `weight_scale`, producing wrong values (~500x off). Since proj_in/proj_out process every token entering/leaving the transformer, this corrupts the entire output.

**The fix** (from the handoff doc, not yet implemented): Make `_move_non_blocks_to_device` FP8-aware — preserve FP8 dtype when moving to GPU, don't cast to bf16:

```python
if preserve_fp8 and _is_fp8_dtype(p.dtype):
    p.data = p.data.to(device, non_blocking=True)  # move only, keep FP8
else:
    p.data = p.data.to(device, dtype=torch.bfloat16, non_blocking=True)
```

**Two paths exist today**:
1. `dequant_bf16` (default) — re-reads FP8+scale from checkpoint, writes correct bf16 values. Works but doubles memory (22GB FP8 → 35GB bf16), forcing Stagehand
2. `fp8_hooks` — keeps FP8 weights, dequants per-layer at forward time. Fits in VRAM but has grid artifacts due to the `_move_non_blocks_to_device` bug

**Workaround in production**: FP8 requests on 24GB GPUs get rerouted to the distilled bf16 checkpoint (`ltx-2.3-22b-distilled.safetensors`). This was disabled in commit 399b35c but the reroute function `_should_prefer_desktop_fast_ltx_checkpoint` still exists.

## New EriQuant Features (uncommitted)

Three new weight tensor types in `eriquant/eriquant/tensor/weights/`:
- `fp8_rowwise.py` — `FP8RowwiseWeightTensor`: PyTorch tensor subclass that stores FP8 data + per-row scales, auto-dequants on `F.linear`, tries `_scaled_mm` on SM89+
- `mxfp8/` — MX FP8 block format
- `nvfp4/` — NVIDIA FP4 format

`FP8RowwiseWeightTensor` could replace both the forward hooks AND the full dequant — it keeps weights as FP8 in memory (~14GB) while providing correct dequant at forward time as a tensor subclass (transparent to the model).

## File Locations

| File | What |
|------|------|
| `serenityflow/bridge/serenity_api.py` | All changes — FP8 Gemma loader, meta tensor fix, dtype fix |
| `serenityflow/bridge/fp8_dequant.py` | Standalone FP8 dequant utilities |
| `~/.serenity/models/text_encoders/gemma-3-12b-it-fp8/` | FP8 Gemma model dir |
| `/home/alex/serenity/output/HANDOFF_FP8_GRID_ARTIFACTS.md` | Detailed grid artifact analysis |
| `/home/alex/serenityflow-v2/worked_on_fp8.md` | Previous FP8 session handoff |
| `/home/alex/serenity/eriquant/eriquant/tensor/weights/fp8_rowwise.py` | New EriQuant FP8 tensor subclass |

## Test Commands

```bash
# FP8 Gemma generation (works, has grid artifacts from transformer dequant)
cd /home/alex/serenityflow-v2 && .venv/bin/python -c "
import os, torch, logging
logging.basicConfig(level=logging.INFO, format='%(name)s %(message)s')
from serenityflow.bridge.serenity_api import load_ltxv_model, sample_ltxv
CHECKPOINT = os.path.expanduser('~/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors')
model = load_ltxv_model(checkpoint_path=CHECKPOINT, gemma_path='', backend='auto')
result = sample_ltxv(model=model, prompt='A golden retriever running through a sunlit meadow', width=512, height=320, num_frames=97, steps=8, guidance_scale=1.0, stg_scale=0.0, seed=42, frame_rate=25.0, mode='distilled')
print(f'Video shape: {result[\"video\"].shape}')
"

# Run tests
cd /home/alex/serenityflow-v2 && .venv/bin/python -m pytest tests/test_fp8_dequant.py tests/test_ltx_gemma_compat.py -q
```

## What To Do Next

1. **Fix the grid artifacts**: Apply the `_move_non_blocks_to_device` fix from the handoff doc, OR replace the forward hook path with EriQuant's `FP8RowwiseWeightTensor` subclass
2. **Test FP8 hooks after fix**: Run generation with `SERENITY_LTX_SCALED_FP8_BACKEND=fp8_hooks` and compare output quality against bf16 dequant
3. **Benchmark**: FP8 hooks should be ~3.5x faster than bf16 dequant+Stagehand
