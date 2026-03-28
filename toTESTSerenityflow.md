# SerenityFlow GPU Tests — Deferred

Blocked by GPU being in use. Run these when GPU is free.

## Real-ESRGAN Upscale (V9)

### Basic Function
- [ ] 2x upscale of 720p clip produces 1440p output
- [ ] 4x upscale of 720p clip produces 2880p output
- [ ] Audio preserved in output
- [ ] Output replaces source_path in project (matches CodeFormer pattern)
- [ ] Preview endpoint returns JPEG with visible quality improvement
- [ ] Progress WS events (`esrgan_progress`, `esrgan_complete`) fire correctly
- [ ] Output frame count matches input exactly
- [ ] Output fps matches input fps

### Tiled Inference
- [ ] tile=512 produces no visible seam artifacts
- [ ] tile=256 produces no visible seam artifacts
- [ ] tile=0 (whole frame) works on 720p input
- [ ] Tile boundary pixels are seamless (pad=10 sufficient)
- [ ] FP16 doesn't produce artifacts on high-contrast edges

### Edge Cases
- [ ] Very small input (128x128) doesn't crash
- [ ] Odd-dimension input (721x481) handled correctly
- [ ] Cancel mid-processing cleans up temp + partial output files
- [ ] Missing model file returns clear error (not crash)
- [ ] 4x upscale of 1080p doesn't OOM with tile=512 on 24GB card
- [ ] Concurrent requests properly blocked (409 response)

### VRAM & Disk
- [ ] Model loads/unloads cleanly — no VRAM leak across multiple runs
- [ ] No temp video files left in /tmp after success or failure
- [ ] Disk space: large output files created without silent corruption

### UI
- [ ] Context menu shows "Upscale (Real-ESRGAN)..." for video clips
- [ ] Dialog shows correct source dimensions
- [ ] Output dimensions update live when toggling 2x/4x radio
- [ ] Quality dropdown works (Fast/Balanced/Safe)
- [ ] Preview button fetches and displays single frame
- [ ] Progress overlay shows blue bar with "Upscale X%" on clip
- [ ] Badge shows "2×" or "4×" after completion
- [ ] Status endpoint reports model missing when .pth absent

## Deflicker / Temporal Consistency (V10)

### Light Mode (ffmpeg)
- [ ] Light mode produces output with reduced flicker on AI-generated video
- [ ] Window slider (3-15) affects output
- [ ] Completes nearly instantly
- [ ] Audio preserved

### Medium Mode (histogram matching)
- [ ] Normalizes brightness variations across frames
- [ ] Strength=0 returns near-identical output
- [ ] Doesn't wash out colors (LAB L-channel only)
- [ ] Progress events fire correctly
- [ ] Cancel works mid-processing

### Heavy Mode (optical flow)
- [ ] Blends frames smoothly without ghosting at alpha=0.15
- [ ] Doesn't introduce ghosting on fast motion
- [ ] Alpha=0.05 produces subtle smoothing
- [ ] Progress events fire (~2-5 sec/frame for 720p)
- [ ] Cancel works mid-processing

### Edge Cases
- [ ] 2-frame clip doesn't crash in heavy mode
- [ ] Single-frame clip handled gracefully
- [ ] Already-smooth video not degraded by any mode
- [ ] Output frame count matches input exactly
- [ ] Concurrent requests blocked (409)

### UI
- [ ] Context menu shows "Deflicker..." for video clips
- [ ] Mode radio switches visible settings panel
- [ ] Light: window slider, Medium: strength + EMA, Heavy: blend alpha
- [ ] Progress overlay shows green bar with "Deflicker X%"
- [ ] "DF" badge appears after completion
- [ ] Badge persists after page reload

## Audio Enhancement (V11)

### ffmpeg Presets
- [ ] All 5 presets produce audibly different output
- [ ] "Clean Speech" noticeably reduces background noise
- [ ] "Normalize" brings quiet audio to standard loudness
- [ ] "De-hum" removes electrical hum
- [ ] "Podcast" compresses voice for consistent levels
- [ ] Video stream untouched (c:v copy — no re-encoding)
- [ ] Audio preview (5s) plays in browser correctly

### DeepFilterNet (if installed)
- [ ] AI enhancement removes background noise from speech
- [ ] Progress events fire (1/3, 2/3, 3/3)
- [ ] Not-installed state shows helpful error with pip command

### Edge Cases
- [ ] Clip with no audio stream returns clear error
- [ ] Very short clip (< 1 second) doesn't crash
- [ ] Stereo audio handled correctly by ffmpeg presets
- [ ] Cancel works during processing
- [ ] Concurrent requests blocked (409)

### Quality
- [ ] No audio/video sync drift after processing
- [ ] Audio duration exactly matches original
- [ ] No clipping introduced by normalization
- [ ] Output file size reasonable (AAC 192k)

### UI
- [ ] Context menu shows "Enhance Audio..." for video clips
- [ ] Mode toggle switches between ffmpeg presets and DeepFilter
- [ ] Preset dropdown updates description text
- [ ] Preview 5s button plays audio sample
- [ ] Progress overlay shows purple bar with "Audio X%"
- [ ] Music note badge appears after completion
- [ ] Badge persists after page reload
- [ ] Presets endpoint returns all presets with descriptions

## CodeFormer Face Restore (V8) — Untested from Prior Session

### Quick Smoke Test
- [ ] Right-click → Restore Faces... opens dialog
- [ ] Fidelity slider works, preview returns restored frame
- [ ] Full clip processing completes and replaces source_path
- [ ] FACE badge appears after completion
