# SerenityFlow Generate Tab — Full Visual Overhaul Plan

## Status: PLANNED (not yet implemented)

## Reference
- InvokeAI screenshots at `/home/alex/Pictures/Screenshots/Screenshot from 2026-03-14 21-07-29.png`
- InvokeAI source at `~/invokeai-ref/invokeai/frontend/web/src/`
- Our Generate tab: `serenityflow/canvas/js/generate.js` + `css/generate.css`

## What Needs to Change

### 1. Top Toolbar (NEW — between topbar and content)
Add a horizontal toolbar strip at the top of the center panel:
```
[Gen btn] [1▲▼] | [📷] [🖌] [≡] [✕] |  [info] | [↩] [↪] [↔] [↕] [⭐] [📑] [🗑] | [✕]
```
- Generate button (yellow/accent, with spark icon) + batch count with up/down
- View mode icons: image viewer, brush, list, close
- Center: info/metadata
- Right: undo, redo, flip horizontal, flip vertical, star/favorite, copy, delete
- Most of these are stubs initially — just the visual structure

### 2. Left Panel — Image Section Overhaul
Replace current aspect buttons + bare number inputs with:
```
Image                                          ▲
┌──────────────────────────────────────────────┐
│ Aspect [1:1 ▾]  [↔] [🔒] [✦]   ┌────────┐ │
│ Width  [────●──────────] [1024]  │        │ │
│ Height [────●──────────] [1024]  │  1:1   │ │
│ Seed   [0         ] [⟳] Random ●│        │ │
│            ↕ Advanced Options    └────────┘ │
└──────────────────────────────────────────────┘
```
- Aspect: dropdown selector (not button grid)
- Swap dimensions button (↔)
- Lock aspect ratio (🔒)
- Optimal size button (✦) — sets best size for current model
- Width/Height: label + slider + number input (all on one row)
- Visual preview box: small rectangle showing current aspect ratio (right side)
- Seed: input + shuffle + Random toggle switch (blue toggle, not button)
- "Advanced Options" collapsible sub-section

### 3. Left Panel — Generation Section Overhaul
```
Generation  [EPICREALISM...] [SDXL]           ▲
┌──────────────────────────────────────────────┐
│ Model  [epicrealismXL_v6...  ▾] [✦] [⚙]    │
│ Concepts  [No LoRAs Installed        ▾]     │
│            ↕ Advanced Options                │
│   Steps [────●──────] [20]                   │
│   CFG   [────●──────] [7.0]                  │
│   Sampler [euler ▾]                          │
└──────────────────────────────────────────────┘
```
- Section header shows model name badge + arch badge (colored: blue=SDXL, etc.)
- Model picker with refresh + settings icons
- LoRA/Concepts as a dropdown (not button + hidden select)
- Advanced Options sub-section with steps/CFG/sampler inside

### 4. Left Panel — Additional Sections (stubs)
- **Compositing** (collapsible, empty) — for future inpaint compositing settings
- **Refiner** (collapsible, SDXL only) — for SDXL refiner model
- **Advanced** (collapsible) — VAE picker, clip skip, CFG rescale, seamless tiling

### 5. Right Panel — Gallery Overhaul
Replace bare "GALLERY" header + grid with:
```
┌──────────────────────────┐
│ [Layers] [Gallery]  [✕]  │ ← tab switcher
├──────────────────────────┤
│ Hide Boards         ▲    │
│ 📁 Uncategorized  AUTO   │
├──────────────────────────┤
│ [Images] [Assets] [⬆][⚙][🔍] │ ← sub-tabs + actions
├──────────────────────────┤
│ [thumb] [thumb] [thumb]  │
│ [thumb] [thumb] [thumb]  │
│ < [1] >  Jump ▾          │ ← pagination
└──────────────────────────┘
```
- Layers/Gallery tab switcher at top
- Boards section (collapsible)
- Images/Assets tab switcher
- Upload button, settings popover, search
- Pagination at bottom (or infinite scroll)

### 6. Floating Side Toolbar (between left panel and center)
Vertical stack of circular icon buttons:
```
[≡]  ← toggle left panel
[✦]  ← generate (green/accent)
[✕]  ← cancel current (red)
[🗑] ← delete current
```

### 7. Prompt Labels
- "Positive Prompt" (not "PROMPT")
- `</>` icon button for dynamic prompts/syntax
- `{}` icon button for prompt templates

### 8. CSS Polish
- Wider left panel (380px like InvokeAI, not 280px)
- Input fields: more padding, softer borders
- Section headers: proper accordion with chevron icon (not just text)
- Sliders: custom styled with accent color
- Toggle switches: rounded pill style for Random seed

## Implementation Priority
1. Width/Height sliders + visual preview box (biggest UX gap)
2. Top toolbar with action icons
3. Gallery tabs + search
4. Model badge + arch badge
5. Floating side toolbar
6. Additional collapsible sections
7. Prompt label polish

## Files to Modify
- `canvas/js/generate.js` — restructure buildLeftHTML, buildCenterHTML, buildRightHTML
- `canvas/css/generate.css` — extensive CSS additions
- `canvas/index.html` — may need toolbar HTML

## DO NOT MODIFY
- shell.js, ws.js, api.js, workflow-builder.js, canvas-tab.js, simple.js, queue.js, models.js, settings.js
- Any Python files
