# SerenityFlow Canvas Tab ↔ InvokeAI Node Editor Parity Plan

## Gap Analysis

### What SerenityFlow HAS that works well
| Feature | Status | Notes |
|---------|--------|-------|
| Node create/delete/duplicate | ✅ | Double-click search, sidebar drag, context menu |
| Typed connections | ✅ | 15 types with color coding, compatibility validation |
| Bezier wires | ✅ | Type-colored, hit-area for deletion |
| Inline widgets | ✅ | INT, FLOAT, STRING, BOOLEAN, COMBO — drag scrub, overlays |
| Copy/paste/duplicate | ✅ | Ctrl+C/V/D with connection remapping |
| Undo/redo | ✅ | 50-state stack, Ctrl+Z/Y |
| Zoom/pan | ✅ | Scroll wheel zoom toward cursor, drag pan, grid |
| Box select + multi-select | ✅ | Ctrl+click, rubber band, Ctrl+A |
| Node search palette | ✅ | Double-click or right-click, instant filter |
| Sidebar node library | ✅ | Categories, search, drag-to-canvas |
| Properties panel | ✅ | Double-click node, all widget types editable |
| Workflow save/load | ✅ | 3 format support (native, ComfyUI, raw API) |
| Execution highlighting | ✅ | Orange/green/red node borders, WS-driven |
| Queue from editor | ✅ | Space/Ctrl+Enter, interrupt button |
| Auto-save | ✅ | Every 5s to sessionStorage |
| Output preview | ✅ | Floating preview panel |
| Toast notifications | ✅ | Success/error with auto-dismiss |

### GAPS — What InvokeAI has that SerenityFlow doesn't

#### HIGH PRIORITY (Core editor features)

| # | Feature | InvokeAI | SerenityFlow | Impact |
|---|---------|----------|-------------|--------|
| 1 | **Minimap** | ✅ Toggleable viewport overview | ❌ None | **HIGH** — essential for large workflows |
| 2 | **Snap to grid** | ✅ 25px grid, toggleable | ❌ Grid is visual only | **HIGH** — cleaner layouts |
| 3 | **Auto-layout** | ✅ LR/TB direction, spacing controls, layering strategy | ❌ None | **HIGH** — huge QoL for messy graphs |
| 4 | **Node collapse/expand** | ✅ Toggle to hide ports/widgets | ❌ None | **HIGH** — reduces visual clutter |
| 5 | **Fit view** | ✅ Button + keyboard, animated | ❌ No zoom-to-fit | **HIGH** — basic navigation |
| 6 | **Edge animation** | ✅ Toggleable dash animation during execution | ❌ Static wires | **MED** — execution feedback |
| 7 | **Notes/annotation nodes** | ✅ Free-form text nodes for documentation | ❌ None | **MED** — workflow documentation |

#### MEDIUM PRIORITY (Polish & UX)

| # | Feature | InvokeAI | SerenityFlow | Impact |
|---|---------|----------|-------------|--------|
| 8 | **Edge reconnect** | ✅ Drag edge from one port to another | ❌ Must delete + recreate | **MED** — faster rewiring |
| 9 | **Viewport controls panel** | ✅ Zoom in/out/fit buttons in corner | ❌ Scroll only | **MED** — discoverability |
| 10 | **Collapsed edges** | ✅ Show count badge when both nodes collapsed | ❌ N/A (no collapse) | **MED** — depends on #4 |
| 11 | **Color-coded edges** | ✅ Toggle: edges colored by field type | ⚠️ Already colored | Minor — already done |
| 12 | **Node status badges** | ✅ Pending/progress %/completed/failed icons | ⚠️ Border color only | **MED** — richer feedback |
| 13 | **Execution progress %** | ✅ Circular progress badge per node | ❌ No per-node progress | **MED** — user feedback |
| 14 | **Workflow metadata** | ✅ Name, author, description, tags, version | ❌ Just filename | **MED** — organization |
| 15 | **Settings toggles** | ✅ Edge animation, edge colors, snap, validate, node opacity | ❌ None | **MED** — customization |

#### LOWER PRIORITY (Advanced features)

| # | Feature | InvokeAI | SerenityFlow | Impact |
|---|---------|----------|-------------|--------|
| 16 | **Workflow library** | ✅ Save/load/manage multiple workflows | ❌ Single file save/load | **LOW** — nice to have |
| 17 | **Inspector tabs** | ✅ 4 tabs: details, outputs, data, template | ⚠️ Single properties panel | **LOW** — our panel works |
| 18 | **Batch field grouping** | ✅ Visual batch group colors | ❌ None | **LOW** — advanced use case |
| 19 | **Generator widgets** | ✅ Float/Int/String/Image generators | ❌ Basic widgets only | **LOW** — ComfyUI doesn't have these either |
| 20 | **Collection widgets** | ✅ Multi-value list editors | ❌ None | **LOW** — specialized |
| 21 | **Color picker widget** | ✅ Native color field | ❌ None | **LOW** — rarely needed |
| 22 | **Image field widget** | ✅ Image preview + selection | ❌ None | **LOW** — specialized |
| 23 | **Node builder/editor mode** | ✅ Form field customization UI | ❌ None | **LOW** — meta-feature |

---

## Implementation Phases

### Phase 1: Navigation & Layout (HIGH PRIORITY)
**Scope**: Core viewport features that InvokeAI users expect
**Files**: `canvas.js`, `canvas.css`

1. **Minimap** — Small viewport-in-viewport overlay (bottom-right corner)
   - Shows all nodes as colored rectangles proportional to real size
   - Current viewport shown as a semi-transparent rect
   - Click/drag minimap to navigate
   - Toggle button to show/hide

2. **Fit View** — Zoom/pan to show all nodes
   - Calculate bounding box of all nodes
   - Set scale and position to fit with padding
   - Animate transition (300ms ease)
   - Keyboard shortcut: `H` (home) or `Ctrl+0`

3. **Viewport Controls Panel** — Button cluster in bottom-left
   - Zoom in (+), Zoom out (-), Fit view, Toggle minimap
   - Small floating panel with icon buttons
   - Same style as InvokeAI's viewport controls

4. **Snap to Grid** — Magnetic grid alignment during drag
   - Grid size: 20px (matches existing visual grid)
   - Round node position to nearest grid point on drag end
   - Toggle via settings or keyboard shortcut (`Shift+G`)
   - Visual grid already exists, just add snap logic

### Phase 2: Node Collapse & Auto-Layout (HIGH PRIORITY)
**Scope**: Node management features for complex workflows
**Files**: `node.js`, `canvas.js`, `canvas.css`

5. **Node Collapse/Expand** — Toggle to minimize node
   - Collapse button (chevron) in node header
   - Collapsed: show only header bar (title + type badge)
   - Ports still visible as dots on collapsed header edges
   - Connections remain, just shorter beziers
   - State: `node.collapsed` boolean
   - Double-click header to toggle

6. **Auto-Layout** — Automatic node arrangement
   - Topological sort of node graph
   - Left-to-right layout (outputs flow right)
   - Configurable spacing (node gap: 60px horizontal, 40px vertical)
   - Fit view after layout
   - Button in viewport controls + keyboard shortcut (`Ctrl+L`)
   - Simple layered layout algorithm (no external dependency)

### Phase 3: Execution Feedback (MEDIUM PRIORITY)
**Scope**: Richer execution visualization
**Files**: `node.js`, `toolbar.js`, `connection.js`, `canvas.css`

7. **Edge Animation** — Animated dashes during execution
   - When workflow running: all edges get animated dash pattern
   - CSS animation on Konva line dash
   - Toggle in settings
   - Stop animation when execution completes

8. **Node Status Badges** — Icon badges showing execution state
   - Top-right corner of node: small colored circle
   - Pending: gray dot
   - Running: pulsing accent dot + step count
   - Completed: green checkmark
   - Failed: red X
   - Progress updates from WS `progress` events

9. **Notes Nodes** — Free-form text annotation nodes
   - Special node type: "Note" (not in backend, purely frontend)
   - Yellow/amber header color
   - Multiline text area body
   - No ports (not connectable)
   - Resizable (drag corner)
   - Add via right-click menu or sidebar

### Phase 4: Edge & Connection Polish (MEDIUM PRIORITY)
**Scope**: Connection management improvements
**Files**: `port.js`, `connection.js`, `canvas.js`

10. **Edge Reconnect** — Drag existing connection to new port
    - Click on connected input port to "pick up" the edge
    - Drag to new compatible input
    - Old connection removed, new one created
    - Type validation during drag

11. **Workflow Metadata Panel** — Editable workflow info
    - Name (shown in topbar, editable)
    - Author, description, tags
    - Saved with workflow JSON
    - Small popover from workflow name click

### Phase 5: Settings & Polish (LOWER PRIORITY)
**Scope**: Editor customization
**Files**: `canvas.js`, new `canvas-settings.js`

12. **Canvas Settings Popover** — Toggleable editor options
    - Snap to grid (on/off)
    - Edge animation (on/off)
    - Show minimap (on/off)
    - Grid visible (on/off)
    - Node opacity slider (0.5–1.0)
    - Persist to localStorage

13. **Workflow Library** — Save/manage multiple workflows
    - localStorage-backed workflow list
    - Save current, load from list, delete, rename
    - Simple list panel (slide-in from left or modal)

### DEFERRED (Future)
- Batch field grouping / batch processing
- Generator/collection widgets
- Color picker / image field widgets
- Node builder/form editor mode
- Lasso selection mode
- Inspector with multiple tabs (outputs, raw data, template)
- Collapsed edge count badges

---

## Files Modified Per Phase

| Phase | Files | Estimated Lines |
|-------|-------|----------------|
| 1 | canvas.js, canvas.css | +250 |
| 2 | node.js, canvas.js, canvas.css | +350 |
| 3 | node.js, connection.js, toolbar.js, canvas.css | +300 |
| 4 | port.js, connection.js, canvas.js, workflow.js | +200 |
| 5 | canvas.js (or new), canvas.css | +250 |
| **Total** | | **~1,350 lines** |

---

## Execution Strategy

Phases 1–2 are the highest-impact changes — minimap, fit view, snap, collapse, and auto-layout transform the editor from "functional" to "professional".

Phase 3 adds execution polish that makes the editor feel alive during generation.

Phases 4–5 are quality-of-life improvements that can be done incrementally.
