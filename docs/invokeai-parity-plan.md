# SerenityFlow ↔ InvokeAI Feature Parity Plan

## Gap Analysis (What InvokeAI has that SerenityFlow doesn't)

### GENERATE TAB GAPS

#### Left Panel — Prompts
| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Positive prompt | ✅ Textarea with autocomplete, prompt assist | ✅ Textarea with auto-grow, token count | Minor — missing autocomplete/assist |
| Negative prompt | ✅ Toggleable, with assist | ✅ Always shown (hidden for flux/video) | Minor — no toggle button |
| Prompt history (↑/↓) | ✅ Navigate with Alt+↑/↓ | ❌ None | **GAP** |
| Dynamic prompts syntax | ✅ Full wildcard/conditional expansion | ❌ Stub buttons only | **GAP** (complex, defer) |
| Style presets | ✅ Full preset system with preview | ✅ Simple mode only (12 styles) | **GAP** — bring to Generate tab |
| Prompt templates | ✅ Template selection | ❌ Stub button only | **GAP** |
| Prompt attention (Ctrl+↑/↓) | ✅ Bracket weight adjustment | ❌ None | **GAP** |

#### Left Panel — Image Settings
| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Aspect ratios | Free,21:9,16:9,3:2,4:3,1:1,3:4,2:3,9:16,9:21 | 1:1,4:3,16:9,3:4,9:16 | **GAP** — add Free,21:9,3:2,2:3,9:21 |
| Width slider | 64–1536 (slider), 64–4096 (input), step 64/8 | 256–2048 (slider), 256–4096 (input), step 64 | Minor — expand range |
| Height slider | Same as width | Same as width | Minor |
| Swap button | ✅ | ✅ | OK |
| Lock aspect | ✅ | ✅ | OK |
| Optimal size | ✅ Model-aware defaults | ✅ | OK |
| Visual preview | ✅ 108x108 box | ✅ 60x60 box | Minor — enlarge |
| Seed | 0–4,294,967,295, disabled when Random ON | ✅ -1 = random | Minor |
| Seed shuffle | ✅ Disabled when Random ON | ✅ | OK |

#### Left Panel — Generation Settings
| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Model picker (searchable) | ✅ Grouped by base, search, badges, size | ✅ Basic dropdown, arch badge | **GAP** — add search, grouping |
| LoRA enable/disable toggle | ✅ Per-LoRA toggle | ❌ Only add/remove | **GAP** |
| LoRA weight range | -1 to 2 (slider), -10 to 10 (input) | 0 to 2 (slider) | **GAP** — expand range |
| Steps range | 1–100 (slider), 1–500 (input) | 1–150 | Minor |
| CFG range | 1–20 (slider), 1–200 (input) | 1–20 | Minor |
| Guidance range | 1–6 (slider), 1–20 (input) | 1–10 | Minor |
| Scheduler options | 30 options (all Karras variants) | 8 options | **GAP** — add more schedulers |
| FLUX DyPE preset | ✅ Off/Manual/Auto/Area/4K | ❌ None | **GAP** (advanced, defer) |
| Batch count | ✅ | ✅ | OK |

#### Left Panel — Compositing
| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Coherence mode | ✅ Gaussian/Box/Staged | ❌ Stub | **GAP** |
| Edge size slider | ✅ 0–128 | ❌ Stub | **GAP** |
| Min denoise | ✅ 0–1 | ❌ Stub | **GAP** |
| Mask blur | ✅ 0–128 | ❌ Stub | **GAP** |
| Infill methods | ✅ patchmatch/lama/cv2/color/tile | ❌ Stub | **GAP** |

#### Left Panel — Refiner (SDXL)
| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Refiner model picker | ✅ | ❌ Stub | **GAP** |
| Refiner scheduler | ✅ | ❌ Stub | **GAP** |
| Refiner steps/CFG | ✅ | ❌ Stub | **GAP** |
| Refiner start (0–1) | ✅ | ❌ Stub | **GAP** |
| Aesthetic scores | ✅ ±1–10 | ❌ Stub | **GAP** |

#### Left Panel — Advanced
| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| VAE picker | ✅ Searchable dropdown | ❌ Stub | **GAP** |
| VAE precision (fp16/fp32) | ✅ | ❌ Stub | **GAP** |
| CLIP skip (0–12) | ✅ SD1.5 only | ❌ Stub | **GAP** |
| CFG rescale (0–0.99) | ✅ | ❌ Stub | **GAP** |
| Seamless tiling X/Y | ✅ | ❌ Stub | **GAP** |

### GALLERY GAPS

| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Layers/Gallery tabs | ✅ | ✅ Visual only | Minor |
| Board creation/deletion | ✅ Full CRUD | ❌ Stub | **GAP** |
| Board auto-add | ✅ | ❌ None | **GAP** |
| Image search | ✅ Metadata search | ❌ Stub | **GAP** |
| Multi-select (Ctrl/Shift+click) | ✅ | ❌ None | **GAP** |
| Star/favorite images | ✅ | ❌ None | **GAP** |
| Image context menu (right-click) | ✅ 15+ actions | ❌ None | **GAP** |
| Image comparison (slider/side-by-side) | ✅ 3 modes | ❌ None | **GAP** (advanced, defer) |
| Gallery settings popover | ✅ 6 settings | ❌ Stub | **GAP** |
| Pagination controls | ✅ Page buttons + jump | ❌ Stub | **GAP** |
| Thumbnail size slider | ✅ 45–256px | ❌ Fixed 2-col | **GAP** |
| Bulk operations | ✅ Star/delete/download/move | ❌ None | **GAP** |
| Metadata viewer | ✅ 5 tabs | ❌ None | **GAP** |
| Image upload | ✅ Button + drag-drop | ❌ Stub | **GAP** |

### TOOLBAR GAPS

| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Top toolbar | ✅ Full action bar | ✅ Partial | Minor |
| Floating left toolbar | ✅ Invoke + cancel + clear | ✅ Generate + cancel + delete | OK |
| Floating right toolbar (gallery toggle) | ✅ | ✅ Via top toolbar | OK |
| Keyboard shortcuts | ✅ 60+ hotkeys | ❌ None | **GAP** |
| Undo/redo | ✅ Ctrl+Z/Y | ❌ Stub buttons | **GAP** |
| Panel toggle hotkeys (T, G, F) | ✅ | ❌ None | **GAP** |

### QUEUE TAB GAPS

| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Queue list | ✅ Virtualized, expandable | ✅ Basic list | Minor |
| Job cancel | ✅ Per-item cancel | ❌ Missing | **GAP** |
| Job retry/requeue | ✅ | ❌ None | **GAP** |
| Queue pause/resume | ✅ | ❌ None | **GAP** |
| Execution time display | ✅ Per-item | ✅ Elapsed timer | OK |
| Queue statistics | ✅ In-progress/pending/completed/failed | ❌ None | **GAP** |

### MODELS TAB GAPS

| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Model list + filter | ✅ Search, type filter, badges | ✅ Basic search + filter | Minor |
| Model details view | ✅ Full metadata grid | ❌ None | **GAP** |
| Model download/install | ✅ 5 methods (URL, HF, scan, etc.) | ❌ None | **GAP** (complex, defer) |
| Model delete | ✅ With confirmation | ❌ None | **GAP** |
| Model convert (ckpt→diffusers) | ✅ | ❌ None | **GAP** (defer) |
| Bulk model operations | ✅ | ❌ None | **GAP** |
| Missing model detection | ✅ | ❌ None | **GAP** |

### SETTINGS TAB GAPS

| Feature | InvokeAI | SerenityFlow | Gap |
|---------|----------|-------------|-----|
| Theme/accent | ✅ | ✅ | OK |
| Default mode/tab | ✅ | ✅ | OK |
| Output format/quality | ✅ | ✅ | OK |
| Language selector | ✅ i18n | ❌ None | **GAP** (defer) |
| NSFW checker toggle | ✅ | ❌ None | N/A |
| Developer logging | ✅ Level + namespace filter | ❌ None | **GAP** |
| Clear intermediates | ✅ | ❌ None | **GAP** |
| Reset WebUI | ✅ | ✅ Reset to defaults | OK |
| Confirm on delete | ✅ | ✅ | OK |

---

## Implementation Phases

### Phase 1: Generate Tab — Controls Upgrade (HIGH PRIORITY)
**Scope**: Make left panel controls match InvokeAI precision
- Add missing aspect ratios (Free, 21:9, 3:2, 2:3, 9:21)
- Expand scheduler list (add all Karras variants, ~22 more options)
- LoRA enable/disable toggle per card
- LoRA weight range expand to -1..2 slider, -10..10 input
- Prompt history (store last 20 prompts, navigate with ↑/↓)
- Style presets in Generate tab (port from Simple mode)
- Expand slider ranges (steps 1–500 input, CFG 1–200 input)
- Searchable model picker with base-model grouping
- Prompt attention weight adjustment (Ctrl+↑/↓ on selected word)

### Phase 2: Gallery — Full Feature Build (HIGH PRIORITY)
**Scope**: Bring gallery to InvokeAI parity
- Image starring (favorite toggle per image)
- Multi-select (Ctrl+click, Shift+click range select)
- Right-click context menu (download, delete, star, use prompt, use seed, use all params, copy)
- Gallery search (filter by prompt text stored in gallery metadata)
- Thumbnail size slider (45–256px via settings popover)
- Gallery settings popover (size, sort direction, starred first, auto-switch)
- Pagination controls (page buttons, jump-to-page)
- Image metadata panel (show prompt, seed, model, steps, CFG for selected image)
- Store metadata with gallery items (prompt, model, seed, arch, dimensions)

### Phase 3: Advanced Left Panel Sections (MEDIUM PRIORITY)
**Scope**: Fill in stub sections with real controls
- **Advanced section**: VAE picker dropdown (fetch from /object_info), CLIP skip slider (0–12, SD1.5 only), CFG rescale slider (0–0.99), Seamless tiling X/Y toggles
- **Refiner section**: Refiner model picker, scheduler, steps, CFG, start slider (0–1), aesthetic scores (SDXL only, conditional visibility)
- **Compositing section**: Coherence mode dropdown, edge size slider, min denoise slider, mask blur slider, infill method dropdown with method-specific sub-options

### Phase 4: Queue Tab Upgrade (MEDIUM PRIORITY)
**Scope**: Queue management parity
- Per-item cancel button
- Queue statistics bar (pending, running, completed, failed counts)
- Job retry/requeue button
- Expanded item details (show prompt, model, dimensions, seed)
- Queue pause/resume processor

### Phase 5: Keyboard Shortcuts (MEDIUM PRIORITY)
**Scope**: Full keyboard navigation
- Global: Ctrl+Enter (invoke), 1-6 (tab switch), T/O (toggle left), G (toggle right), F (toggle all)
- Generate: Alt+↑/↓ (prompt history), Ctrl+↑/↓ (attention weight)
- Gallery: Arrow keys (navigate), Delete (delete), . (star), Ctrl+A (select all), Esc (clear selection)
- Undo/Redo: Ctrl+Z / Ctrl+Shift+Z (for gallery operations)

### Phase 6: Models Tab Upgrade (LOWER PRIORITY)
**Scope**: Model management
- Model details panel (click model → show metadata: format, size, path, base, type)
- Model delete with confirmation
- Missing model detection and indicator
- Bulk delete (checkbox + delete selected)

### Phase 7: Image Upload & Use (LOWER PRIORITY)
**Scope**: Image input features
- Gallery upload button (file picker for images)
- Drag-and-drop upload to gallery
- "Use prompt" from gallery image (populate prompt fields)
- "Use seed" from gallery image
- "Use all parameters" recall from metadata
- Download selected image

### DEFERRED (Future phases, complex features)
- Dynamic prompts engine (wildcard/conditional syntax)
- Image comparison viewer (slider/side-by-side/hover modes)
- Model download/install from URL/HuggingFace
- Model conversion (checkpoint → diffusers)
- Control layers / IP-Adapter / ControlNet UI
- Canvas inpainting tools
- Regional prompts
- Board system with full CRUD + drag-and-drop
- i18n / language selection
- FLUX DyPE presets

---

## Files Modified Per Phase

| Phase | Files | Estimated Lines |
|-------|-------|----------------|
| 1 | generate.js, generate.css | +300 |
| 2 | generate.js, generate.css | +500 |
| 3 | generate.js, generate.css | +400 |
| 4 | queue.js, queue.css (new?) | +300 |
| 5 | generate.js, shell.js | +200 |
| 6 | models.js, models.css | +250 |
| 7 | generate.js, generate.css | +200 |

## Execution Strategy

Each phase gets 3 agents:
1. **Builder** — Implements all features for the phase
2. **Bug Fixer** — Reviews for broken functionality, null refs, missing bindings, edge cases
3. **Skeptic** — Aesthetic review: does it look like InvokeAI? CSS issues? Consistency?

After all phases: **Verification agent** confirms full parity against the audit.
