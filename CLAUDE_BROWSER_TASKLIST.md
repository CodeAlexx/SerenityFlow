# SerenityFlow NLE — Test Tasklist

All features from commit f1e9480. UNTESTED — needs GPU + running SerenityFlow server.

## Setup
- Start SerenityFlow server
- Open NLE in browser
- Have a test video clip loaded in a project (ideally with audio, 720p, 5+ seconds)
- Have a second clip with NO audio for edge case testing

---

## V8: CodeFormer Face Restore

### Smoke Test
- [ ] Right-click video clip → context menu shows "Restore Faces..."
- [ ] Click it → dialog opens with fidelity slider (0-1, default 0.7)
- [ ] License note visible in dialog (S-Lab 1.0 non-commercial)
- [ ] Preview button returns restored face thumbnail
- [ ] Click Start → progress overlay appears on clip (coral bar)
- [ ] Processing completes → FACE badge appears on clip
- [ ] source_path updated in project (check Properties)
- [ ] Badge persists after page reload

### Edge Cases
- [ ] Missing model files → clear error message with path
- [ ] Clip with no faces → processes without crash, frames pass through

---

## V9: Real-ESRGAN Upscale

### Basic Function
- [ ] Right-click clip → "Upscale (Real-ESRGAN)..." in context menu
- [ ] Dialog opens with 2x/4x radio, quality dropdown, preview button
- [ ] Select 2x → output dims update in dialog
- [ ] Select 4x → output dims update in dialog
- [ ] Quality dropdown has Fast/Balanced/Safe options
- [ ] Preview button shows upscaled single frame
- [ ] Click Start with 2x → progress overlay (blue bar, "Upscale X%")
- [ ] Completes → badge shows "2×" on clip
- [ ] source_path updated in project
- [ ] Audio preserved in output

### Quality Checks
- [ ] 2x of 720p → output is 1440p (check with ffprobe)
- [ ] 4x of 720p → output is 2880p
- [ ] No visible tile seams at default tile=512
- [ ] Output frame count matches input
- [ ] Output fps matches input

### Edge Cases
- [ ] Missing model → clear error with path to place .pth
- [ ] Cancel mid-processing → temp files cleaned up
- [ ] Second request while processing → 409 "already processing"
- [ ] Badge persists after page reload

---

## V10: Deflicker

### Light Mode
- [ ] Right-click clip → "Deflicker..." in context menu
- [ ] Dialog opens with Light/Medium/Heavy radio buttons
- [ ] Light selected by default → window slider visible (3-15)
- [ ] Click Start → completes quickly (ffmpeg filter, near-instant)
- [ ] Output has reduced flicker vs original
- [ ] Audio preserved
- [ ] "DF" badge appears (green)

### Medium Mode
- [ ] Select Medium → strength slider + EMA decay slider appear
- [ ] Light settings hidden
- [ ] Click Start → progress bar shows "Deflicker X%" (green)
- [ ] Progress updates smoothly
- [ ] Output normalizes brightness variations
- [ ] Cancel works mid-processing
- [ ] Colors not washed out (LAB L-channel only)

### Heavy Mode
- [ ] Select Heavy → blend alpha slider appears (0.05-0.3)
- [ ] Click Start → slower processing with progress updates
- [ ] Output is smooth without ghosting at alpha=0.15
- [ ] Cancel works mid-processing

### Edge Cases
- [ ] 2-frame clip in heavy mode → no crash
- [ ] Second request while processing → 409
- [ ] Badge persists after page reload
- [ ] Output frame count matches input

---

## V11: Audio Enhancement

### ffmpeg Presets
- [ ] Right-click clip → "Enhance Audio..." in context menu
- [ ] Dialog opens → ffmpeg Presets mode selected by default
- [ ] Preset dropdown has: Normalize, Clean Speech, Podcast, Music, De-hum
- [ ] Changing preset updates description text below dropdown
- [ ] "Preview 5s" button → audio plays in browser (5 second sample)
- [ ] Click Apply with "Normalize" → processing starts
- [ ] Completes → music note badge appears (purple)
- [ ] source_path updated in project
- [ ] Video stream NOT re-encoded (c:v copy — file size similar except audio)

### Preset Quality
- [ ] Normalize: quiet audio brought to standard loudness
- [ ] Clean Speech: background noise reduced, voice clearer
- [ ] Podcast: voice levels more consistent/compressed
- [ ] De-hum: electrical hum removed (test with humming audio)
- [ ] Music: gentle normalization, less aggressive than Normalize

### DeepFilter AI Mode
- [ ] Toggle to "AI (DeepFilter)" mode
- [ ] If not installed → "(not installed)" shown with pip command
- [ ] If installed → description shown, Apply button works
- [ ] Progress events fire (33%, 66%, 100%)
- [ ] Output removes background noise while preserving voice

### Edge Cases
- [ ] Clip with NO audio → clear error "No audio stream found"
- [ ] Cancel during processing works
- [ ] Second request while processing → 409
- [ ] Badge persists after page reload
- [ ] No audio/video sync drift (compare timestamps with ffprobe)
- [ ] Audio duration matches original exactly

---

## Cross-Feature Checks

### Badge Stacking
- [ ] Apply multiple features to same clip (e.g., upscale + deflicker + audio)
- [ ] All badges visible side by side (no overlap)
- [ ] Each badge has correct color (FACE=coral, 2×=blue, DF=green, ♪=purple)

### Badge Visibility During Processing
- [ ] While clip A is processing, clip B's existing badges still show
- [ ] Only the actively-processing clip's badge is hidden

### Context Menu Order
- [ ] All items present: Interpolate 2×, Interpolate 4×, Restore Faces, Upscale, Deflicker, Enhance Audio
- [ ] Only shown for video clips with source_path

### Concurrent Blocking
- [ ] Start ESRGAN on clip A → try Deflicker on clip B → 409 for deflicker (separate job queues, should work independently)
- [ ] Start two ESRGAN jobs → second gets 409

### Project Persistence
- [ ] Process a clip → reload page → processed source_path and badges survive
- [ ] Export still works with processed clips
