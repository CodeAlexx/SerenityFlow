# Test: CodeFormer Face Restoration — SerenityFlow NLE

## Setup

- Open SerenityFlow NLE at **http://127.0.0.1:8188**
- Click the **video editor icon** (4th icon in left sidebar, looks like a film strip/clapperboard)
- You should see the NLE timeline with tracks: Video 1, Video 2, Audio, Subtitles, Video 3
- Video 1 has two clips: "Intro Renamed" (blue placeholder) and "tokyo_10s" (real video frames)
- **Use the "tokyo_10s" clip** for all face restore tests — it has actual video content
- Note: models may not be installed. Many tests below verify error handling for that case.

---

## Test 1: Status Endpoint

**Action:** Open browser DevTools (F12) → Console tab. Run:
```js
fetch('/video_edit/facetools/status').then(r => r.json()).then(d => console.log(d))
```

**Expected response shape:**
```json
{
  "available": true or false,
  "models": { "codeformer": true or false },
  "model_dir": "/some/path/models/facetools",
  "processing": false
}
```

**Verify:**
- [ ] Response is 200
- [ ] `available` is a boolean
- [ ] `models` is an object with `codeformer` key
- [ ] `model_dir` is a string path
- [ ] `processing` is `false` (nothing running yet)

---

## Test 2: Context Menu Shows "Restore Faces..."

**Action:** Right-click on the **"tokyo_10s"** clip in Video 1 track.

**Verify:**
- [ ] Context menu appears
- [ ] Menu contains "Restore Faces..." option
- [ ] Menu also shows: Interpolate 2x (RIFE), Interpolate 4x (RIFE), Upscale (Real-ESRGAN)..., Deflicker..., Enhance Audio...
- [ ] "Restore Faces..." is NOT shown for subtitle clips or audio clips (right-click "Hello world" or "Music.mp3" to confirm)

---

## Test 3: Missing Models Error

**Prerequisite:** Status endpoint shows `"available": false` (codeformer.pth not installed).

**Action:** Right-click "tokyo_10s" → click "Restore Faces..."

**Verify:**
- [ ] An alert/dialog appears saying models are not found
- [ ] Alert shows which model is missing (e.g. "codeformer")
- [ ] Alert shows the model_dir path where to place files
- [ ] No crash, no unhandled error in console

---

## Test 4: Face Restore Dialog (if models ARE present)

**Prerequisite:** `codeformer.pth` exists in the model_dir shown by status endpoint.

**Action:** Right-click "tokyo_10s" → click "Restore Faces..."

**Verify:**
- [ ] Dialog appears titled "Restore Faces (CodeFormer)"
- [ ] Fidelity slider present, range 0 to 1, step 0.05, default value **0.70**
- [ ] Slider label updates as you drag (shows current value like "0.70")
- [ ] Hint text: left says "Higher quality", right says "Higher fidelity"
- [ ] License notice visible: "Non-commercial license (S-Lab 1.0)"
- [ ] Three buttons: **Preview**, **Cancel**, **Start**
- [ ] X close button in top-right of dialog
- [ ] Clicking Cancel closes the dialog
- [ ] Clicking X closes the dialog

---

## Test 5: Preview Button

**Prerequisite:** Models installed, dialog open.

**Action:** Set fidelity slider to 0.5, click **Preview**.

**Verify:**
- [ ] Preview button text changes to "Loading..." and becomes disabled
- [ ] After a moment, a preview image appears in the dialog
- [ ] Preview button text returns to "Preview" and re-enables
- [ ] Changing fidelity slider and clicking Preview again shows a new preview
- [ ] If preview fails, an alert shows "Preview failed: ..." (not a crash)

---

## Test 6: Start Processing

**Prerequisite:** Models installed, dialog open.

**Action:** Set fidelity to 0.7, click **Start**.

**Verify:**
- [ ] Dialog closes immediately
- [ ] The "tokyo_10s" clip on the timeline shows a progress overlay
- [ ] Progress text updates: "Restoring faces... X%"
- [ ] Progress increases from 0% to 100%
- [ ] On completion, progress overlay disappears
- [ ] The clip's source file is updated (clip now points to `*_facefix.mp4`)

**Console verification (run during processing):**
```js
// Check WS events are firing
SerenityWS.on('face_restore_progress', d => console.log('progress', d));
SerenityWS.on('face_restore_complete', d => console.log('complete', d));
SerenityWS.on('face_restore_error', d => console.log('error', d));
```

- [ ] `face_restore_progress` events have: `job_id`, `frame`, `total`, `percent`, `clip_id`
- [ ] `face_restore_complete` event has: `job_id`, `output_path`, `clip_id`, `fidelity`
- [ ] `percent` goes from 0 to 100

---

## Test 7: Audio Preserved

**Action:** After face restore completes on a clip that has audio.

**Verify:**
- [ ] Play the restored clip — audio is present and synced
- [ ] Original audio was not lost during face restoration

---

## Test 8: Fidelity w=0 (Maximum Quality)

**Action:** Start face restore with fidelity slider at **0.00**.

**Verify:**
- [ ] Processing completes without error
- [ ] Result shows maximum face quality enhancement (faces may look different from original — this is expected at w=0)

---

## Test 9: Fidelity w=1 (Maximum Fidelity)

**Action:** Start face restore with fidelity slider at **1.00**.

**Verify:**
- [ ] Processing completes without error
- [ ] Result shows minimal change from original (high fidelity = preserves input)

---

## Test 10: Cancel Mid-Processing

**Prerequisite:** Start a face restore job.

**Action:** While progress is showing, look for a way to cancel (or run in console):
```js
// Get the active job ID from the progress event, then:
fetch('/video_edit/facetools/cancel/JOB_ID_HERE', { method: 'POST' })
  .then(r => r.json()).then(d => console.log(d))
```

**Verify:**
- [ ] Cancel returns `{ "status": "cancelling" }`
- [ ] Progress overlay disappears
- [ ] A `face_restore_error` event fires with `"error": "Cancelled"`
- [ ] No temp files left behind (check model_dir for stray .mp4 files)
- [ ] No crash in console

---

## Test 11: Concurrent Request Blocked

**Action:** Start face restore on "tokyo_10s". While processing, try to start another:
```js
fetch('/video_edit/facetools/restore', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ project_id: 'YOUR_PROJECT_ID', clip_id: 'tokyo_10s_clip_id', fidelity: 0.7 })
}).then(r => r.json()).then(d => console.log(d))
```

**Verify:**
- [ ] Second request returns 409 status
- [ ] Error message says "already processing another clip"

---

## Test 12: Frames With No Faces

**Action:** Run face restore on a clip that contains NO faces (e.g. landscape footage, or the "Intro Renamed" placeholder clip if it has video content).

**Verify:**
- [ ] Processing completes successfully (does not crash)
- [ ] Frames without faces pass through unchanged
- [ ] Output video looks identical to input

---

## Test 13: VRAM Cleanup

**Action:** Run face restore, let it complete. Check GPU memory:
```js
fetch('/system_stats').then(r => r.json()).then(d => console.log(d.devices))
```

**Verify:**
- [ ] After completion, GPU VRAM usage returns close to pre-processing levels
- [ ] No VRAM leak (model was unloaded)
- [ ] Running face restore a second time also works (model reloads cleanly)

---

## Test 14: Export Still Works

**Action:** After face-restoring a clip, click **Export** (top-right of timeline).

**Verify:**
- [ ] Export completes without error
- [ ] Exported video includes the face-restored clip
- [ ] Face restoration is visible in the export

---

## Test 15: Non-Video Clips Don't Show Option

**Action:** Right-click on non-video elements:
- Right-click "Music.mp3" (Audio track)
- Right-click "Hello world" (Subtitles track)
- Right-click "Overlay" (Video 3 — but this may be a video)

**Verify:**
- [ ] "Restore Faces..." does NOT appear for audio clips
- [ ] "Restore Faces..." does NOT appear for subtitle clips
- [ ] It only appears for clips with a `source_path` pointing to a video file
