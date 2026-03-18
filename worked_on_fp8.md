# FP8 Handoff

Date: 2026-03-17
Repo: `/home/alex/serenityflow-v2`
Workspace: `/home/alex/serenity`

## Current Status

The old scaled-FP8 LTX path is still not good on this 24 GB card.

What is now working:
- Requests for `ltx-2.3-22b-dev-fp8.safetensors` on a 24 GB class GPU now default to a desktop-style non-FP8 fast path.
- That fast path now prefers `ltx-2.3-22b-distilled.safetensors`.
- Stagehand is active for Gemma and both transformer stages in that path.
- The resulting video quality is much better than the broken FP8 runs and is the first path that looks usable.

What is still broken:
- The experimental scaled-FP8 runtime still produces bad output on this machine.
- The pure FP8/dequant path still has woven/canvas artifacts and smeared detail.
- Official/GPTQ Gemma path still has unresolved builder/meta-tensor issues in other branches of the code.

## Main Code Changes

### 1. Scaled FP8 dequant compatibility

Files:
- `serenityflow/bridge/fp8_dequant.py`
- `serenityflow/bridge/serenity_api.py`

Changes:
- Added `dequantize_fp8(...)` helper.
- Dequant now follows `fp8 -> float32 -> scale multiply -> cast`.
- Scale expansion now handles:
  - scalar
  - per-row `[N, 1]`
  - flat `[N]` repaired to rowwise
  - blockwise repeat-interleave expansion
- `_dequant_scaled_fp8_weights(...)` now uses the shared helper instead of `bf16 * bf16 scale`.

Why:
- This matches the compatibility behavior needed for weight-only FP8 fallback on SM86 hardware.
- Native `_scaled_mm` FP8 compute is not available on this 3090-class GPU.

### 2. 24 GB fast-path reroute

Files:
- `serenityflow/bridge/serenity_api.py`

Changes:
- Added desktop-fast checkpoint reroute helpers.
- If all of the following are true:
  - backend is not `official`
  - checkpoint looks like `ltx-2.3` scaled-FP8
  - checkpoint really has weight scales
  - CUDA is available
  - total VRAM is `<= 26 GiB`
  - `SERENITY_LTX_SCALED_FP8_EXPERIMENTAL` is not set
- Then Serenity reroutes to:
  - `ltx-2.3-22b-distilled.safetensors`
- The lookup is exact-name only so it cannot alias back to the FP8 file by relaxed matching.

Why:
- This matches LTX-Desktop fast behavior more closely than trying to force the broken scaled-FP8 path.

### 3. Stagehand fallback safety

Files:
- `serenityflow/bridge/serenity_api.py`

Changes:
- Added `_module_has_fp8_params(...)`.
- Stage 1 and stage 2 now re-run `_move_non_blocks_to_device(...)` before Stagehand startup after direct-GPU fallback.
- This avoids leaving non-block pieces on the wrong device after a failed direct-GPU attempt.

## Environment Switches

- `SERENITY_LTX_SCALED_FP8_EXPERIMENTAL=1`
  - disables the 24 GB fast-path reroute
  - forces the old scaled-FP8 runtime path
- `SERENITY_LTX_SCALED_FP8_BACKEND=eriquant_fp8`
  - explicit opt-in only
- default backend for scaled checkpoints:
  - `dequant_bf16`

## Test Status

Passed:
- `python3 -m pytest /home/alex/serenityflow-v2/tests/test_fp8_dequant.py -q`
- `python3 -m pytest /home/alex/serenityflow-v2/tests/test_ltx_gemma_compat.py -q`

Coverage added:
- rowwise scale handling
- flat row-scale repair
- blockwise scale expansion
- env override for experimental FP8
- exact non-FP8 reroute behavior
- fake eriquant module injection for isolated tests

## Important Artifacts

### Broken / reference outputs

- Broken scaled-FP8 direct GPU:
  - `/home/alex/serenity/output/ltx_fp8_directgpu_smoke.mp4`
  - `/home/alex/serenity/output/ltx_fp8_directgpu_smoke_frame0.png`
  - `/home/alex/serenity/output/ltx_fp8_directgpu_smoke_frame12.png`
- Broken dequant-FP8 fallback:
  - `/home/alex/serenity/output/ltx_fp8_dequant_auto_smoke_standalone.mp4`
  - `/home/alex/serenity/output/ltx_fp8_dequant_auto_smoke_standalone_frame0.png`
  - `/home/alex/serenity/output/ltx_fp8_dequant_auto_smoke_standalone_frame12.png`

### Better fast-path outputs

- `22b-dev` reroute:
  - `/home/alex/serenity/output/ltx_desktop_fast_auto_smoke.mp4`
  - `/home/alex/serenity/output/ltx_desktop_fast_auto_smoke_frame0.png`
  - `/home/alex/serenity/output/ltx_desktop_fast_auto_smoke_frame12.png`
  - quality: better than FP8, but still too soft and smeared
  - runtime: about `645.5s`

- `22b-distilled` reroute:
  - `/home/alex/serenity/output/ltx_desktop_fast_distilled_smoke.mp4`
  - `/home/alex/serenity/output/ltx_desktop_fast_distilled_smoke_frame0.png`
  - `/home/alex/serenity/output/ltx_desktop_fast_distilled_smoke_frame12.png`
  - `/home/alex/serenity/output/ltx_desktop_fast_distilled_smoke.log`
  - `/home/alex/serenity/output/ltx_desktop_fast_distilled_smoke_meta.json`
  - quality: first run that looks usable
  - runtime: about `402.0s`

## Quality Read

`22b-distilled` fast path:
- woven FP8 artifact is gone
- fence, sidewalk, dust, and interview setup are readable
- subject is recognizable
- still somewhat soft
- still some motion smear
- framing drifted from the original intended shot

So:
- not perfect
- but clearly better than the FP8 path
- good enough to keep as default for now on 24 GB cards

## If Picking This Back Up Later

Start from these questions:

1. Is the user asking for best quality now, or to revive true scaled-FP8 later?
2. If best quality now:
   - keep using the `22b-distilled` fast path
   - compare with prompt/seed tweaks rather than going back to experimental FP8
3. If reviving true scaled-FP8:
   - do not assume Stagehand is the issue
   - it was already ruled out earlier
   - focus on checkpoint/runtime compatibility instead

## Suggested Next Steps

If optimizing the current usable path:
- compare `22b-distilled` with NAG-like behavior from `LTX-Desktop`
- tune stage-2 settings or prompt schedule for less smear
- test a few more seeds/prompts to confirm stability

If returning to experimental FP8 research:
- compare checkpoint internals against upstream expectations
- verify whether any remaining non-linear FP8 tensors need special handling
- do not treat current scaled-FP8 outputs as production ready
