/**
 * Canvas Tab — SerenityFlow Phase 4
 * Konva.js infinite canvas with layers, bounding box, brush/eraser, inpaint/outpaint.
 * Completely separate Konva instance from the Workflows graph editor.
 */

var CanvasTab = (function() {
    'use strict';

    var initialized = false;
    var konvaReady = false;

    // Konva objects
    var stage = null;
    var backgroundLayer = null;
    var rasterLayers = [];
    var uiLayer = null;
    var boundingBox = null;
    var sizeLabel = null;
    var resizeHandles = [];
    var brushCursor = null;
    var bgRect = null;

    // State
    var activeTool = 'select';
    var activeLayerId = null;
    var brushSize = 20;
    var brushColor = '#ffffff';
    var isDrawing = false;
    var currentLine = null;
    var isPanning = false;
    var isSpaceHeld = false;
    var panStart = { x: 0, y: 0 };
    var stageStart = { x: 0, y: 0 };
    var canvasGenerating = false;
    var layerIdCounter = 0;
    var activeHandle = null;
    var handleStartBox = null;
    var handleStartMouse = null;
    var brushHardness = 1;
    var bboxAspectLocked = false;
    var bboxLockedRatio = 1;

    // Generation state
    var genState = {
        model: null,
        prompt: '',
        denoise: 0.75,
        steps: 20,
        cfg: 7.0,
        guidance: 3.5,
        seed: -1,
        arch: 'sd15',
        frames: 97,
        fps: 24
    };

    // DOM refs
    var els = {};

    // Handle size constant
    var HANDLE_SIZE = 12;

    // ── Lucide SVG icons ──
    var ICONS = {
        mousePointer: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/><path d="M13 13l6 6"/></svg>',
        brush: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08"/><path d="M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z"/></svg>',
        eraser: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/></svg>',
        move: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="5 9 2 12 5 15"/><polyline points="9 5 12 2 15 5"/><polyline points="15 19 12 22 9 19"/><polyline points="19 9 22 12 19 15"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="12" y1="2" x2="12" y2="22"/></svg>',
        maximize: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>',
        eye: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>',
        eyeOff: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" y1="2" x2="22" y2="22"/></svg>',
        mask: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18"/><path d="M3 12h18"/></svg>',
        undo: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/></svg>',
        redo: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13"/></svg>',
        trash: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>',
        lock: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
        unlock: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>'
    };

    // ── Helpers ──
    function snapTo64(val) { return Math.max(64, Math.round(val / 64) * 64); }
    function snapTo32(val) { return Math.max(64, Math.round(val / 32) * 32); }
    function clampDim(val) { return Math.max(256, Math.min(4096, val)); }
    function isVideoArch() { return genState.arch === 'ltxv' || genState.arch === 'wan'; }
    function snapDimForArch(val) {
        return isVideoArch() ? snapTo32(val) : snapTo64(val);
    }
    function clampDimForArch(val) {
        if (isVideoArch()) return Math.max(64, Math.min(1280, snapTo32(val)));
        return clampDim(snapTo64(val));
    }

    function getRelativePointerPosition() {
        var transform = stage.getAbsoluteTransform().copy().invert();
        var pos = stage.getPointerPosition();
        if (!pos) return { x: 0, y: 0 };
        return transform.point(pos);
    }

    function getActiveKonvaLayer() {
        for (var i = 0; i < rasterLayers.length; i++) {
            if (rasterLayers[i].id === activeLayerId) return rasterLayers[i].konvaLayer;
        }
        return rasterLayers.length > 0 ? rasterLayers[0].konvaLayer : null;
    }

    function getLayerById(id) {
        for (var i = 0; i < rasterLayers.length; i++) {
            if (rasterLayers[i].id === id) return rasterLayers[i];
        }
        return null;
    }

    // ── Snapshot-based Undo/Redo ──
    var History = (function() {
        var stack = [];
        var cursor = -1;
        var MAX = 50;
        var snapshotting = false; // prevent re-entry

        function snapshot() {
            if (!stage || !boundingBox) return null;
            return {
                layers: rasterLayers.map(function(l) {
                    return {
                        id: l.id, type: l.type, name: l.name,
                        visible: l.visible, locked: l.locked,
                        opacity: l.opacity !== undefined ? l.opacity : 1,
                        imageData: l.konvaLayer.toDataURL()
                    };
                }),
                bbox: {
                    x: boundingBox.x(), y: boundingBox.y(),
                    width: boundingBox.width(), height: boundingBox.height()
                },
                activeLayerId: activeLayerId
            };
        }

        function push() {
            if (snapshotting) return;
            var snap = snapshot();
            if (!snap) return;
            // Discard redo entries ahead of cursor
            stack.splice(cursor + 1);
            stack.push(snap);
            if (stack.length > MAX) stack.shift();
            cursor = stack.length - 1;
            updateButtons();
        }

        function undo() {
            if (cursor <= 0 || snapshotting) return;
            cursor--;
            restore(stack[cursor]);
            updateButtons();
        }

        function redo() {
            if (cursor >= stack.length - 1 || snapshotting) return;
            cursor++;
            restore(stack[cursor]);
            updateButtons();
        }

        function restore(entry) {
            if (!entry || !stage) return;
            snapshotting = true;

            // Remove existing raster layers
            rasterLayers.forEach(function(l) { l.konvaLayer.destroy(); });
            rasterLayers = [];

            // Rebuild layers from snapshot
            var loaded = 0;
            var total = entry.layers.length;

            entry.layers.forEach(function(saved) {
                var konvaLayer = new Konva.Layer();
                stage.add(konvaLayer);
                if (uiLayer && uiLayer.parent) uiLayer.moveToTop();

                var info = {
                    id: saved.id, name: saved.name, type: saved.type,
                    visible: saved.visible, locked: saved.locked,
                    opacity: saved.opacity, konvaLayer: konvaLayer
                };
                konvaLayer.opacity(saved.opacity);
                if (!saved.visible) konvaLayer.hide();
                rasterLayers.push(info);

                // Restore pixel content
                if (saved.imageData && saved.imageData !== 'data:,') {
                    var img = new Image();
                    img.onload = function() {
                        var kImg = new Konva.Image({ image: img, x: 0, y: 0 });
                        konvaLayer.add(kImg);
                        konvaLayer.batchDraw();
                        loaded++;
                        if (loaded >= total) finishRestore(entry);
                    };
                    img.onerror = function() {
                        loaded++;
                        if (loaded >= total) finishRestore(entry);
                    };
                    img.src = saved.imageData;
                } else {
                    loaded++;
                    if (loaded >= total) finishRestore(entry);
                }
            });

            if (total === 0) finishRestore(entry);
        }

        function finishRestore(entry) {
            // Restore bbox
            if (entry.bbox) {
                boundingBox.x(entry.bbox.x);
                boundingBox.y(entry.bbox.y);
                boundingBox.width(entry.bbox.width);
                boundingBox.height(entry.bbox.height);
                updateHandles();
                updateSizeLabel();
            }

            // Restore active layer
            activeLayerId = entry.activeLayerId;
            layerIdCounter = Math.max.apply(null, rasterLayers.map(function(l) { return l.id; }).concat([0]));

            renderLayerList();
            stage.batchDraw();
            snapshotting = false;
        }

        function updateButtons() {
            var undoBtn = document.getElementById('cv-undo');
            var redoBtn = document.getElementById('cv-redo');
            if (undoBtn) {
                undoBtn.disabled = cursor <= 0;
                undoBtn.title = 'Undo (Ctrl+Z)' + (cursor > 0 ? ' (' + cursor + ')' : '');
            }
            if (redoBtn) {
                redoBtn.disabled = cursor >= stack.length - 1;
            }
        }

        return { push: push, undo: undo, redo: redo, updateButtons: updateButtons };
    })();

    // ── Floating Preview Panel ──
    var lastPreviewSrc = null;
    var lastPreviewIsVideo = false;

    function showCanvasPreview(src, isVideo) {
        lastPreviewSrc = src;
        lastPreviewIsVideo = isVideo;
        var body = document.getElementById('canvas-preview-body');
        if (!body) return;
        body.innerHTML = isVideo
            ? '<video src="' + src + '" autoplay loop muted playsinline controls></video>'
            : '<img src="' + src + '">';
        var panel = document.getElementById('canvas-preview-panel');
        if (panel) panel.style.display = 'block';
    }

    function hideCanvasPreview() {
        var panel = document.getElementById('canvas-preview-panel');
        if (panel) panel.style.display = 'none';
        lastPreviewSrc = null;
    }

    function acceptCanvasPreview() {
        if (!lastPreviewSrc) return;
        if (lastPreviewIsVideo) {
            placeVideoOverlayOnCanvas(lastPreviewSrc);
        } else {
            placeResultOnCanvas(lastPreviewSrc);
        }
        hideCanvasPreview();
    }

    function setupPreviewPanel() {
        var panel = document.getElementById('canvas-preview-panel');
        var header = document.getElementById('canvas-preview-header');
        var closeBtn = document.getElementById('canvas-preview-close');
        var acceptBtn = document.getElementById('canvas-preview-accept');
        var discardBtn = document.getElementById('canvas-preview-discard');
        var downloadBtn = document.getElementById('canvas-preview-download');

        if (!panel || !header) return;

        // Drag behavior
        var dragging = false, startX, startY, startLeft, startTop;
        header.addEventListener('mousedown', function(e) {
            dragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startLeft = panel.offsetLeft;
            startTop = panel.offsetTop;
        });
        document.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            var newLeft = Math.max(0, Math.min(startLeft + e.clientX - startX, window.innerWidth - panel.offsetWidth));
            var newTop = Math.max(0, Math.min(startTop + e.clientY - startY, window.innerHeight - 40));
            panel.style.left = newLeft + 'px';
            panel.style.top = newTop + 'px';
            panel.style.right = 'auto';
        });
        document.addEventListener('mouseup', function() { dragging = false; });

        if (closeBtn) closeBtn.addEventListener('click', hideCanvasPreview);
        if (discardBtn) discardBtn.addEventListener('click', hideCanvasPreview);
        if (acceptBtn) acceptBtn.addEventListener('click', acceptCanvasPreview);
        if (downloadBtn) {
            downloadBtn.addEventListener('click', function() {
                if (!lastPreviewSrc) return;
                var a = document.createElement('a');
                a.href = lastPreviewSrc;
                a.download = 'serenityflow_canvas_' + Date.now() + (lastPreviewIsVideo ? '.mp4' : '.png');
                a.click();
            });
        }
    }

    // ── Image upload for Control/IPA layers ──
    function handleLayerImageUpload(layerId, file, wellEl) {
        var reader = new FileReader();
        reader.onload = function(ev) {
            var layer = getLayerById(layerId);
            if (layer) {
                layer.refImageSrc = ev.target.result;
                layer.refImageFile = file;
                if (wellEl) {
                    wellEl.innerHTML = '<img src="' + ev.target.result + '">';
                }
                // Also place on canvas as semi-transparent overlay
                var img = new Image();
                img.onload = function() {
                    layer.konvaLayer.destroyChildren();
                    var kImg = new Konva.Image({
                        image: img,
                        x: boundingBox.x(), y: boundingBox.y(),
                        width: boundingBox.width(), height: boundingBox.height(),
                        opacity: 0.5,
                        draggable: activeTool === 'select'
                    });
                    layer.konvaLayer.add(kImg);
                    layer.konvaLayer.batchDraw();
                    History.push();
                };
                img.src = ev.target.result;
            }
        };
        reader.readAsDataURL(file);
    }

    // ── Checkerboard ──
    function createCheckerboardImage() {
        var size = 16;
        var c = document.createElement('canvas');
        c.width = size * 2;
        c.height = size * 2;
        var ctx = c.getContext('2d');
        ctx.fillStyle = '#181824';
        ctx.fillRect(0, 0, size * 2, size * 2);
        ctx.fillStyle = '#1e1e2e';
        ctx.fillRect(0, 0, size, size);
        ctx.fillRect(size, size, size, size);
        var img = new Image();
        img.src = c.toDataURL();
        return img;
    }

    // ── Build DOM ──
    function buildUI() {
        var panel = document.getElementById('panel-canvas');
        if (!panel) return;
        panel.innerHTML = '';

        var layout = document.createElement('div');
        layout.className = 'cv-layout';

        var left = document.createElement('div');
        left.className = 'cv-left';
        left.innerHTML = buildLeftHTML();
        layout.appendChild(left);

        var center = document.createElement('div');
        center.className = 'cv-center';
        center.innerHTML = '<div id="canvas-stage-container"></div>';

        // Add bbox toolbar to center
        var bboxToolbar = document.createElement('div');
        bboxToolbar.id = 'cv-bbox-toolbar';
        bboxToolbar.className = 'cv-bbox-toolbar';
        bboxToolbar.innerHTML =
            '<input type="number" id="cv-bbox-w" class="cv-bbox-input" min="64" max="4096" step="64" value="1024">' +
            '<span class="cv-bbox-x">&times;</span>' +
            '<input type="number" id="cv-bbox-h" class="cv-bbox-input" min="64" max="4096" step="64" value="1024">' +
            '<span class="cv-bbox-sep">|</span>' +
            '<select id="cv-bbox-snap" class="cv-bbox-snap">' +
                '<option value="32">Snap: 32</option>' +
                '<option value="64" selected>Snap: 64</option>' +
                '<option value="128">Snap: 128</option>' +
            '</select>' +
            '<button id="cv-bbox-reset" class="cv-bbox-btn" title="Reset to 1024x1024">Reset</button>' +
            '<button id="cv-bbox-lock" class="cv-bbox-btn" title="Lock aspect ratio">&#128274;</button>' +
            '<button id="cv-bbox-fit" class="cv-bbox-btn" title="Fit to active layer">Fit</button>';
        center.appendChild(bboxToolbar);

        layout.appendChild(center);

        var right = document.createElement('div');
        right.className = 'cv-right';
        right.innerHTML = buildRightHTML();
        layout.appendChild(right);

        panel.appendChild(layout);
        cacheElements();
    }

    function buildLeftHTML() {
        return '' +
            '<div class="cv-tools">' +
                '<button class="cv-tool-btn active" data-tool="select" title="Select / Move (V)">' + ICONS.mousePointer + '</button>' +
                '<button class="cv-tool-btn" data-tool="brush" title="Brush (B)">' + ICONS.brush + '</button>' +
                '<button class="cv-tool-btn" data-tool="eraser" title="Eraser (E)">' + ICONS.eraser + '</button>' +
                '<button class="cv-tool-btn" data-tool="mask" title="Inpaint Mask (M)">' + ICONS.mask + '</button>' +
                '<button class="cv-tool-btn" data-tool="pan" title="Pan (H)">' + ICONS.move + '</button>' +
                '<button class="cv-tool-btn" data-tool="resetView" title="Reset View (F)">' + ICONS.maximize + '</button>' +
                '<hr class="cv-tool-separator">' +
                '<button class="cv-tool-btn cv-undo-btn" id="cv-undo" title="Undo (Ctrl+Z)">' + ICONS.undo + '</button>' +
                '<button class="cv-tool-btn cv-redo-btn" id="cv-redo" title="Redo (Ctrl+Y)">' + ICONS.redo + '</button>' +
            '</div>' +
            '<div id="cv-mask-actions" class="cv-mask-actions" style="display:none">' +
                '<button class="cv-mask-action-btn" id="cv-mask-fill">Fill All</button>' +
                '<button class="cv-mask-action-btn" id="cv-mask-clear">Clear Mask</button>' +
                '<span class="cv-mask-sep">|</span>' +
                '<span class="cv-mask-label">Opacity</span>' +
                '<input type="range" id="cv-mask-opacity" class="cv-mask-opacity-range" min="0.1" max="1" step="0.05" value="0.6">' +
            '</div>' +
            '<div class="cv-layers">' +
                '<div class="cv-layers-header">' +
                    '<span class="cv-layers-title">Layers</span>' +
                    '<div class="cv-layers-add-wrap">' +
                        '<button class="cv-layers-add" id="cv-layers-add-btn" title="Add layer">+</button>' +
                        '<div id="cv-layer-type-menu" class="cv-layer-type-menu" style="display:none">' +
                            '<div class="cv-layer-type-item" data-type="raster">Raster Layer</div>' +
                            '<div class="cv-layer-type-item" data-type="mask">Inpaint Mask</div>' +
                            '<div class="cv-layer-type-item" data-type="control">Control Layer</div>' +
                            '<div class="cv-layer-type-item" data-type="ipadapter">IP-Adapter</div>' +
                            '<div class="cv-layer-type-item" data-type="regional">Regional Prompt</div>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
                '<div class="cv-layer-list" id="cv-layer-list"></div>' +
            '</div>';
    }

    function buildRightHTML() {
        return '' +
            '<div id="cv-brush-section" class="cv-brush-settings" style="display:none">' +
                '<div class="cv-section-title">Brush</div>' +
                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">Size</span>' +
                    '<input type="range" id="cv-brush-size" class="cv-range" min="1" max="200" value="20">' +
                    '<span id="cv-brush-size-val" class="cv-setting-value">20</span>' +
                '</div>' +
                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">Hardness</span>' +
                    '<input type="range" id="cv-brush-hardness" class="cv-range" min="0" max="1" step="0.05" value="1">' +
                    '<span id="cv-brush-hardness-val" class="cv-setting-value">1.0</span>' +
                '</div>' +
                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">Color</span>' +
                    '<input type="color" id="cv-brush-color" class="cv-color-swatch" value="#ffffff">' +
                '</div>' +
                '<hr class="cv-separator">' +
            '</div>' +

            '<div id="cv-control-panel" class="cv-type-panel" style="display:none">' +
                '<div class="cv-section-title">Control Layer</div>' +
                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">Type</span>' +
                    '<select id="cv-control-type" class="cv-select">' +
                        '<option value="depth">Depth</option>' +
                        '<option value="canny">Canny Edge</option>' +
                        '<option value="pose">Pose</option>' +
                        '<option value="normal">Normal Map</option>' +
                        '<option value="lineart">Lineart</option>' +
                    '</select>' +
                '</div>' +
                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">Weight</span>' +
                    '<input type="range" id="cv-control-weight" class="cv-range" min="0" max="2" step="0.05" value="1">' +
                    '<span id="cv-control-weight-val" class="cv-setting-value">1.00</span>' +
                '</div>' +
                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">Start</span>' +
                    '<input type="range" id="cv-control-start" class="cv-range" min="0" max="1" step="0.05" value="0">' +
                    '<span id="cv-control-start-val" class="cv-setting-value">0.00</span>' +
                '</div>' +
                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">End</span>' +
                    '<input type="range" id="cv-control-end" class="cv-range" min="0" max="1" step="0.05" value="1">' +
                    '<span id="cv-control-end-val" class="cv-setting-value">1.00</span>' +
                '</div>' +
                '<div class="cv-image-well" id="cv-control-well">' +
                    '<span class="cv-image-well-placeholder">Drop image or click to upload</span>' +
                '</div>' +
                '<input type="file" id="cv-control-file" accept="image/*" style="display:none">' +
                '<div class="cv-helper-text">ControlNet nodes will be added when available on the backend.</div>' +
            '</div>' +

            '<div id="cv-ipadapter-panel" class="cv-type-panel" style="display:none">' +
                '<div class="cv-section-title">IP-Adapter</div>' +
                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">Weight</span>' +
                    '<input type="range" id="cv-ipa-weight" class="cv-range" min="0" max="2" step="0.05" value="1">' +
                    '<span id="cv-ipa-weight-val" class="cv-setting-value">1.00</span>' +
                '</div>' +
                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">Method</span>' +
                    '<select id="cv-ipa-method" class="cv-select">' +
                        '<option value="style">Style</option>' +
                        '<option value="composition">Composition</option>' +
                        '<option value="style_composition">Style + Composition</option>' +
                    '</select>' +
                '</div>' +
                '<div class="cv-image-well" id="cv-ipa-well">' +
                    '<span class="cv-image-well-placeholder">Drop image or click to upload</span>' +
                '</div>' +
                '<input type="file" id="cv-ipa-file" accept="image/*" style="display:none">' +
                '<div class="cv-helper-text">IP-Adapter nodes will be added when available on the backend.</div>' +
            '</div>' +

            '<div id="cv-regional-panel" class="cv-type-panel" style="display:none">' +
                '<div class="cv-section-title">Regional Prompt</div>' +
                '<label class="cv-setting-label">Region Prompt</label>' +
                '<textarea id="cv-regional-prompt" class="cv-textarea" rows="3" placeholder="Prompt for this region..."></textarea>' +
                '<label class="cv-setting-label" style="margin-top:8px">Negative</label>' +
                '<textarea id="cv-regional-neg" class="cv-textarea" rows="2" placeholder="Negative for this region..."></textarea>' +
                '<div class="cv-helper-text">Draw the region on the canvas. Regional conditioning will be applied when supported.</div>' +
            '</div>' +

            '<div class="cv-gen-settings">' +
                '<div class="cv-section-title">Generation</div>' +

                '<label class="cv-setting-label" style="margin-bottom:2px">Prompt</label>' +
                '<textarea id="cv-prompt" class="cv-textarea" rows="3" placeholder="Describe the content..."></textarea>' +

                '<div class="cv-setting-row" style="margin-top:4px">' +
                    '<span class="cv-setting-label">Denoise</span>' +
                    '<input type="range" id="cv-denoise" class="cv-range" min="0" max="1" step="0.01" value="0.75">' +
                    '<span id="cv-denoise-val" class="cv-setting-value">0.75</span>' +
                '</div>' +
                '<div class="cv-helper-text">Low = subtle changes &middot; High = full reimagining</div>' +

                '<div class="cv-setting-row">' +
                    '<span class="cv-setting-label">Steps</span>' +
                    '<input type="number" id="cv-steps" class="cv-number-input" min="1" max="150" value="20">' +
                    '<input type="range" id="cv-steps-range" class="cv-range" min="1" max="150" value="20">' +
                '</div>' +

                '<div id="cv-cfg-row" class="cv-setting-row">' +
                    '<span class="cv-setting-label">CFG</span>' +
                    '<input type="number" id="cv-cfg" class="cv-number-input" min="1" max="20" step="0.5" value="7.0">' +
                    '<input type="range" id="cv-cfg-range" class="cv-range" min="1" max="20" step="0.5" value="7.0">' +
                '</div>' +

                '<div id="cv-guidance-row" class="cv-setting-row" style="display:none">' +
                    '<span class="cv-setting-label">Guidance</span>' +
                    '<input type="number" id="cv-guidance" class="cv-number-input" min="1" max="10" step="0.5" value="3.5">' +
                    '<input type="range" id="cv-guidance-range" class="cv-range" min="1" max="10" step="0.5" value="3.5">' +
                '</div>' +

                '<div id="cv-video-section" style="display:none">' +
                    '<div class="cv-section-title" style="margin-top:8px">Video</div>' +
                    '<div class="cv-setting-row">' +
                        '<span class="cv-setting-label">Frames</span>' +
                        '<input type="number" id="cv-frames" class="cv-number-input" min="9" max="257" step="8" value="97">' +
                        '<input type="range" id="cv-frames-range" class="cv-range" min="9" max="257" step="8" value="97">' +
                    '</div>' +
                    '<div class="cv-setting-row">' +
                        '<span class="cv-setting-label">FPS</span>' +
                        '<input type="number" id="cv-fps" class="cv-number-input" min="8" max="60" value="24">' +
                        '<input type="range" id="cv-fps-range" class="cv-range" min="8" max="60" value="24">' +
                    '</div>' +
                    '<div id="cv-duration-hint" class="cv-duration-hint"></div>' +
                '</div>' +

                '<label class="cv-setting-label" style="margin-top:4px">Model</label>' +
                '<select id="cv-model" class="cv-select"><option disabled selected>Loading models...</option></select>' +

                '<hr class="cv-separator">' +

                '<button id="cv-import-btn" class="cv-import-btn">Import Image</button>' +
                '<input type="file" id="cv-import-file" accept="image/*" style="display:none">' +

                '<button id="cv-invoke-btn" class="cv-invoke-btn">Invoke</button>' +

                '<div id="cv-progress" class="cv-progress"><div id="cv-progress-bar" class="cv-progress-bar"></div></div>' +
                '<div id="cv-progress-label" class="cv-progress-label"></div>' +
                '<div id="cv-error-banner" class="cv-error-banner"></div>' +
            '</div>';
    }

    function cacheElements() {
        els.layerList = document.getElementById('cv-layer-list');
        els.brushSection = document.getElementById('cv-brush-section');
        els.brushSizeInput = document.getElementById('cv-brush-size');
        els.brushSizeVal = document.getElementById('cv-brush-size-val');
        els.brushColorInput = document.getElementById('cv-brush-color');
        els.prompt = document.getElementById('cv-prompt');
        els.denoise = document.getElementById('cv-denoise');
        els.denoiseVal = document.getElementById('cv-denoise-val');
        els.steps = document.getElementById('cv-steps');
        els.stepsRange = document.getElementById('cv-steps-range');
        els.cfgRow = document.getElementById('cv-cfg-row');
        els.cfg = document.getElementById('cv-cfg');
        els.cfgRange = document.getElementById('cv-cfg-range');
        els.guidanceRow = document.getElementById('cv-guidance-row');
        els.guidance = document.getElementById('cv-guidance');
        els.guidanceRange = document.getElementById('cv-guidance-range');
        els.videoSection = document.getElementById('cv-video-section');
        els.framesInput = document.getElementById('cv-frames');
        els.framesRange = document.getElementById('cv-frames-range');
        els.fpsInput = document.getElementById('cv-fps');
        els.fpsRange = document.getElementById('cv-fps-range');
        els.durationHint = document.getElementById('cv-duration-hint');
        els.model = document.getElementById('cv-model');
        els.importBtn = document.getElementById('cv-import-btn');
        els.importFile = document.getElementById('cv-import-file');
        els.invokeBtn = document.getElementById('cv-invoke-btn');
        els.progress = document.getElementById('cv-progress');
        els.progressBar = document.getElementById('cv-progress-bar');
        els.progressLabel = document.getElementById('cv-progress-label');
        els.errorBanner = document.getElementById('cv-error-banner');
    }

    // ── Konva Stage ──
    function initKonva() {
        var container = document.getElementById('canvas-stage-container');
        if (!container) return;

        var w = container.offsetWidth;
        var h = container.offsetHeight;

        // Bail if layout hasn't happened yet — resize() will fix it
        if (w < 100 || h < 100) { w = 800; h = 600; }

        stage = new Konva.Stage({
            container: 'canvas-stage-container',
            width: w,
            height: h,
            draggable: false
        });

        // Background layer
        backgroundLayer = new Konva.Layer({ listening: false });
        stage.add(backgroundLayer);

        var checkerImg = createCheckerboardImage();
        checkerImg.onload = function() {
            bgRect = new Konva.Rect({
                x: -10000, y: -10000,
                width: 20000, height: 20000,
                fillPatternImage: checkerImg,
                fillPatternRepeat: 'repeat',
                listening: false
            });
            backgroundLayer.add(bgRect);
            backgroundLayer.batchDraw();
        };

        // UI layer (will be moved to top after raster layers)
        uiLayer = new Konva.Layer();

        // Initial raster layer
        addLayer('Raster Layer', 'raster');

        // UI layer on top
        stage.add(uiLayer);

        // Bounding box
        initBoundingBox(w, h);

        // Brush cursor
        brushCursor = new Konva.Circle({
            x: 0, y: 0,
            radius: brushSize / 2,
            stroke: '#6c6af5',
            strokeWidth: 1.5,
            visible: false,
            listening: false
        });
        uiLayer.add(brushCursor);

        setupStageEvents();
        uiLayer.batchDraw();
        konvaReady = true;
    }

    // ── Bounding Box ──
    function initBoundingBox(stageW, stageH) {
        var bw = 1024;
        var bh = 1024;
        var cx = Math.round(stageW / 2 - bw / 2);
        var cy = Math.round(stageH / 2 - bh / 2);

        boundingBox = new Konva.Rect({
            x: cx, y: cy,
            width: bw, height: bh,
            stroke: '#6c6af5',
            strokeWidth: 2,
            dash: [10, 5],
            fill: 'rgba(108,106,245,0.05)',
            draggable: true,
            name: 'bounding-box',
            shadowColor: '#6c6af5',
            shadowBlur: 6,
            shadowOpacity: 0.3,
            shadowEnabled: true
        });
        uiLayer.add(boundingBox);

        sizeLabel = new Konva.Text({
            x: cx, y: cy + bh + 8,
            text: bw + ' \u00d7 ' + bh,
            fontSize: 12,
            fontFamily: 'Inter, system-ui, sans-serif',
            fill: '#6c6af5',
            opacity: 0.9,
            listening: false,
            padding: 2
        });
        uiLayer.add(sizeLabel);

        boundingBox.on('dragmove', function() {
            updateHandles();
            updateSizeLabel();
            updateVideoOverlayPosition();
        });

        boundingBox.on('dragend', function() {
            History.push();
        });

        createResizeHandles();
    }

    function createResizeHandles() {
        resizeHandles.forEach(function(h) { h.destroy(); });
        resizeHandles = [];

        var positions = [
            { name: 'tl', cursor: 'nwse-resize' },
            { name: 'tc', cursor: 'ns-resize' },
            { name: 'tr', cursor: 'nesw-resize' },
            { name: 'ml', cursor: 'ew-resize' },
            { name: 'mr', cursor: 'ew-resize' },
            { name: 'bl', cursor: 'nesw-resize' },
            { name: 'bc', cursor: 'ns-resize' },
            { name: 'br', cursor: 'nwse-resize' }
        ];

        var hs = HANDLE_SIZE;
        var half = hs / 2;

        positions.forEach(function(p) {
            var handle = new Konva.Rect({
                width: hs, height: hs,
                fill: '#6c6af5',
                stroke: '#ffffff',
                strokeWidth: 1.5,
                cornerRadius: 2,
                draggable: true,
                name: 'handle-' + p.name
            });

            handle.on('mouseenter', function() {
                stage.container().style.cursor = p.cursor;
            });
            handle.on('mouseleave', function() {
                if (!activeHandle) updateCursor();
            });

            handle.on('dragstart', function() {
                activeHandle = p.name;
                handleStartBox = {
                    x: boundingBox.x(), y: boundingBox.y(),
                    w: boundingBox.width(), h: boundingBox.height()
                };
                handleStartMouse = getRelativePointerPosition();
            });

            handle.on('dragmove', function() {
                handle.position(handle.position());

                var pos = getRelativePointerPosition();
                var dx = pos.x - handleStartMouse.x;
                var dy = pos.y - handleStartMouse.y;

                var newX = handleStartBox.x;
                var newY = handleStartBox.y;
                var newW = handleStartBox.w;
                var newH = handleStartBox.h;

                var nm = activeHandle;
                if (nm.indexOf('l') >= 0) { newX = handleStartBox.x + dx; newW = handleStartBox.w - dx; }
                else if (nm.indexOf('r') >= 0) { newW = handleStartBox.w + dx; }
                if (nm.indexOf('t') >= 0) { newY = handleStartBox.y + dy; newH = handleStartBox.h - dy; }
                else if (nm.indexOf('b') >= 0) { newH = handleStartBox.h + dy; }

                newW = clampDimForArch(newW);
                newH = clampDimForArch(newH);

                if (nm.indexOf('l') >= 0) newX = handleStartBox.x + handleStartBox.w - newW;
                if (nm.indexOf('t') >= 0) newY = handleStartBox.y + handleStartBox.h - newH;

                boundingBox.x(newX);
                boundingBox.y(newY);
                boundingBox.width(newW);
                boundingBox.height(newH);

                updateHandles();
                updateSizeLabel();
                updateVideoOverlayPosition();
            });

            handle.on('dragend', function() {
                activeHandle = null;
                handleStartBox = null;
                handleStartMouse = null;
                updateCursor();
                History.push();
            });

            uiLayer.add(handle);
            resizeHandles.push(handle);
        });

        updateHandles();
    }

    function updateHandles() {
        if (!boundingBox) return;
        var bx = boundingBox.x();
        var by = boundingBox.y();
        var bw = boundingBox.width();
        var bh = boundingBox.height();
        var half = HANDLE_SIZE / 2;

        var coords = [
            { x: bx - half,          y: by - half },           // tl
            { x: bx + bw / 2 - half, y: by - half },           // tc
            { x: bx + bw - half,     y: by - half },           // tr
            { x: bx - half,          y: by + bh / 2 - half },  // ml
            { x: bx + bw - half,     y: by + bh / 2 - half },  // mr
            { x: bx - half,          y: by + bh - half },      // bl
            { x: bx + bw / 2 - half, y: by + bh - half },      // bc
            { x: bx + bw - half,     y: by + bh - half }       // br
        ];

        resizeHandles.forEach(function(h, i) {
            h.x(coords[i].x);
            h.y(coords[i].y);
        });
    }

    function updateSizeLabel() {
        if (!sizeLabel || !boundingBox) return;
        var bw = boundingBox.width();
        var bh = boundingBox.height();
        var label = bw + ' \u00d7 ' + bh;
        if (isVideoArch()) {
            label += ' \u00b7 ' + genState.frames + 'f';
        }
        sizeLabel.text(label);
        // Center label under bbox
        sizeLabel.x(boundingBox.x() + bw / 2 - sizeLabel.width() / 2);
        sizeLabel.y(boundingBox.y() + bh + 8);
        uiLayer.batchDraw();
        updateBboxInputs();
    }

    function updateBboxInputs() {
        var bboxW = document.getElementById('cv-bbox-w');
        var bboxH = document.getElementById('cv-bbox-h');
        if (bboxW && boundingBox) bboxW.value = Math.round(boundingBox.width());
        if (bboxH && boundingBox) bboxH.value = Math.round(boundingBox.height());
    }

    // ── Stage Events ──
    function setupStageEvents() {
        var container = stage.container();

        // Pan: middle mouse OR space+drag OR pan tool
        container.addEventListener('mousedown', function(e) {
            if (e.button === 1 || (e.button === 0 && (isSpaceHeld || activeTool === 'pan'))) {
                isPanning = true;
                panStart.x = e.clientX;
                panStart.y = e.clientY;
                stageStart.x = stage.x();
                stageStart.y = stage.y();
                container.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });

        container.addEventListener('mousemove', function(e) {
            if (isPanning) {
                stage.position({
                    x: stageStart.x + (e.clientX - panStart.x),
                    y: stageStart.y + (e.clientY - panStart.y)
                });
                stage.batchDraw();
                updateVideoOverlayPosition();
            }
        });

        container.addEventListener('mouseup', function() {
            if (isPanning) {
                isPanning = false;
                updateCursor();
            }
        });

        container.addEventListener('auxclick', function(e) {
            if (e.button === 1) e.preventDefault();
        });

        // Zoom
        stage.on('wheel', function(e) {
            e.evt.preventDefault();
            var scaleBy = 1.08;
            var oldScale = stage.scaleX();
            var pointer = stage.getPointerPosition();
            if (!pointer) return;
            var newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;
            var clampedScale = Math.min(Math.max(newScale, 0.1), 10);

            var mousePointTo = {
                x: (pointer.x - stage.x()) / oldScale,
                y: (pointer.y - stage.y()) / oldScale
            };
            stage.scale({ x: clampedScale, y: clampedScale });
            stage.position({
                x: pointer.x - mousePointTo.x * clampedScale,
                y: pointer.y - mousePointTo.y * clampedScale
            });
            stage.batchDraw();
            updateVideoOverlayPosition();
        });

        // Drawing
        stage.on('mousedown', function(e) {
            if (activeTool !== 'brush' && activeTool !== 'eraser' && activeTool !== 'mask') return;
            var activeLayer = getLayerById(activeLayerId);
            if (activeLayer && activeLayer.locked) return;
            if (isPanning || isSpaceHeld) return;
            var target = e.target;
            if (target && target.name() && (target.name() === 'bounding-box' || target.name().indexOf('handle-') === 0)) return;

            isDrawing = true;
            var pos = getRelativePointerPosition();
            var strokeColor = activeTool === 'mask' ? 'rgba(239, 68, 68, 0.5)' : (activeTool === 'eraser' ? '#000' : brushColor);
            var compositeOp = activeTool === 'eraser' ? 'destination-out' : 'source-over';
            currentLine = new Konva.Line({
                stroke: strokeColor,
                strokeWidth: brushSize,
                globalCompositeOperation: compositeOp,
                lineCap: 'round',
                lineJoin: 'round',
                opacity: brushHardness,
                points: [pos.x, pos.y, pos.x, pos.y],
                listening: false
            });
            var layer = getActiveKonvaLayer();
            if (layer) layer.add(currentLine);
        });

        stage.on('mousemove', function() {
            if ((activeTool === 'brush' || activeTool === 'eraser' || activeTool === 'mask') && brushCursor) {
                var pos = getRelativePointerPosition();
                brushCursor.x(pos.x);
                brushCursor.y(pos.y);
                brushCursor.radius(brushSize / 2 / stage.scaleX());
                brushCursor.strokeWidth(1.5 / stage.scaleX());
                if (!brushCursor.visible()) brushCursor.visible(true);
                uiLayer.batchDraw();
            }

            if (!isDrawing || !currentLine) return;
            var pos = getRelativePointerPosition();
            currentLine.points(currentLine.points().concat([pos.x, pos.y]));
            var layer = getActiveKonvaLayer();
            if (layer) layer.batchDraw();
        });

        stage.on('mouseup', function() {
            if (isDrawing && currentLine) {
                History.push();
            }
            isDrawing = false;
            currentLine = null;
        });

        stage.on('mouseleave', function() {
            isDrawing = false;
            currentLine = null;
            if (brushCursor) {
                brushCursor.visible(false);
                uiLayer.batchDraw();
            }
        });

        // Drag and drop
        container.addEventListener('dragover', function(e) { e.preventDefault(); });
        container.addEventListener('drop', function(e) {
            e.preventDefault();
            var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (!file || !file.type.startsWith('image/')) return;
            loadImageFile(file);
        });
    }

    // ── Image loading ──
    function loadImageFile(file) {
        var reader = new FileReader();
        reader.onload = function(ev) {
            var img = new Image();
            img.onload = function() {
                var kImg = new Konva.Image({
                    image: img,
                    x: boundingBox.x(),
                    y: boundingBox.y(),
                    width: boundingBox.width(),
                    height: boundingBox.height(),
                    draggable: activeTool === 'select'
                });
                var layer = getActiveKonvaLayer();
                if (layer) {
                    layer.add(kImg);
                    layer.batchDraw();
                    History.push();
                }
            };
            img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
    }

    // ── Layer Management ──
    function addLayer(name, type) {
        var id = ++layerIdCounter;
        var konvaLayer = new Konva.Layer();
        stage.add(konvaLayer);
        // Keep uiLayer on top
        if (uiLayer && uiLayer.parent) uiLayer.moveToTop();

        var info = { id: id, name: name, type: type || 'raster', visible: true, opacity: 1, locked: false, konvaLayer: konvaLayer };
        rasterLayers.push(info);
        activeLayerId = id;
        renderLayerList();
        History.push();
        return info;
    }

    function renderLayerList() {
        if (!els.layerList) return;
        els.layerList.innerHTML = '';

        for (var i = rasterLayers.length - 1; i >= 0; i--) {
            var layer = rasterLayers[i];
            var row = document.createElement('div');
            row.className = 'cv-layer-row' + (layer.id === activeLayerId ? ' active' : '');
            row.dataset.layerId = layer.id;
            row.draggable = true;

            (function(dragLayer, dragRow) {
                dragRow.addEventListener('dragstart', function(e) {
                    e.dataTransfer.setData('text/plain', String(dragLayer.id));
                    dragRow.classList.add('cv-layer-dragging');
                });
                dragRow.addEventListener('dragend', function() {
                    dragRow.classList.remove('cv-layer-dragging');
                });
                dragRow.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    dragRow.classList.add('cv-layer-dragover');
                });
                dragRow.addEventListener('dragleave', function() {
                    dragRow.classList.remove('cv-layer-dragover');
                });
                dragRow.addEventListener('drop', function(e) {
                    e.preventDefault();
                    dragRow.classList.remove('cv-layer-dragover');
                    var draggedId = parseInt(e.dataTransfer.getData('text/plain'));
                    var targetId = dragLayer.id;
                    if (draggedId === targetId) return;
                    reorderLayer(draggedId, targetId);
                });
            })(layer, row);

            var eyeBtn = document.createElement('button');
            eyeBtn.className = 'cv-layer-eye' + (layer.visible ? '' : ' hidden-layer');
            eyeBtn.innerHTML = layer.visible ? ICONS.eye : ICONS.eyeOff;
            eyeBtn.dataset.layerId = layer.id;

            var badge = document.createElement('span');
            badge.className = 'cv-layer-badge ' + layer.type;
            badge.textContent = layer.type.toUpperCase();

            var nameSpan = document.createElement('span');
            nameSpan.className = 'cv-layer-name';
            nameSpan.textContent = layer.name;

            var deleteBtn = document.createElement('button');
            deleteBtn.className = 'cv-layer-delete';
            deleteBtn.innerHTML = ICONS.trash;
            deleteBtn.dataset.layerId = layer.id;
            deleteBtn.title = 'Delete layer';

            var lockBtn = document.createElement('button');
            lockBtn.className = 'cv-layer-lock' + (layer.locked ? ' locked' : '');
            lockBtn.innerHTML = layer.locked ? ICONS.lock : ICONS.unlock;
            lockBtn.dataset.layerId = layer.id;
            lockBtn.title = layer.locked ? 'Unlock layer' : 'Lock layer';

            var opacitySlider = document.createElement('input');
            opacitySlider.type = 'range';
            opacitySlider.className = 'cv-layer-opacity';
            opacitySlider.min = '0';
            opacitySlider.max = '1';
            opacitySlider.step = '0.05';
            opacitySlider.value = layer.opacity !== undefined ? layer.opacity : 1;
            opacitySlider.title = 'Opacity';
            opacitySlider.dataset.layerId = layer.id;

            nameSpan.addEventListener('dblclick', function(e) {
                e.stopPropagation();
                var thisLayer = layer;
                var input = document.createElement('input');
                input.type = 'text';
                input.className = 'cv-layer-rename-input';
                input.value = thisLayer.name;
                nameSpan.replaceWith(input);
                input.focus();
                input.select();
                function finishRename() {
                    var newName = input.value.trim() || thisLayer.name;
                    thisLayer.name = newName;
                    renderLayerList();
                }
                input.addEventListener('blur', finishRename);
                input.addEventListener('keydown', function(ev) {
                    if (ev.key === 'Enter') { ev.preventDefault(); finishRename(); }
                    if (ev.key === 'Escape') { renderLayerList(); }
                });
            });

            row.appendChild(eyeBtn);
            row.appendChild(lockBtn);
            row.appendChild(badge);
            row.appendChild(nameSpan);
            row.appendChild(opacitySlider);
            row.appendChild(deleteBtn);
            els.layerList.appendChild(row);
        }

        if (isVideoArch()) {
            var frameIndicator = document.createElement('div');
            frameIndicator.className = 'cv-frame-indicator';
            frameIndicator.textContent = genState.frames + ' frames @ ' + genState.fps + 'fps';
            els.layerList.appendChild(frameIndicator);
        }

        els.layerList.onclick = function(e) {
            var delEl = e.target.closest('.cv-layer-delete');
            if (delEl) {
                var delId = parseInt(delEl.dataset.layerId);
                deleteLayer(delId);
                e.stopPropagation();
                return;
            }
            var eyeEl = e.target.closest('.cv-layer-eye');
            if (eyeEl) {
                toggleLayerVisibility(parseInt(eyeEl.dataset.layerId));
                e.stopPropagation();
                return;
            }
            var lockEl = e.target.closest('.cv-layer-lock');
            if (lockEl) {
                var lockId = parseInt(lockEl.dataset.layerId);
                var lyr = getLayerById(lockId);
                if (lyr) {
                    lyr.locked = !lyr.locked;
                    renderLayerList();
                }
                e.stopPropagation();
                return;
            }
            var rowEl = e.target.closest('.cv-layer-row');
            if (rowEl) {
                activeLayerId = parseInt(rowEl.dataset.layerId);
                renderLayerList();
            }
        };

        els.layerList.addEventListener('input', function(e) {
            if (e.target.classList.contains('cv-layer-opacity')) {
                var lid = parseInt(e.target.dataset.layerId);
                var lyr = getLayerById(lid);
                if (lyr) {
                    lyr.opacity = parseFloat(e.target.value);
                    lyr.konvaLayer.opacity(lyr.opacity);
                    lyr.konvaLayer.batchDraw();
                }
            }
        });

        updateMaskActions();
        updateTypePanels();
    }

    function updateMaskActions() {
        var maskActions = document.getElementById('cv-mask-actions');
        if (!maskActions) return;
        var activeLayer = getLayerById(activeLayerId);
        maskActions.style.display = (activeLayer && activeLayer.type === 'mask') ? 'flex' : 'none';
    }

    function updateTypePanels() {
        var controlPanel = document.getElementById('cv-control-panel');
        var ipaPanel = document.getElementById('cv-ipadapter-panel');
        var regionalPanel = document.getElementById('cv-regional-panel');

        var activeLayer = getLayerById(activeLayerId);
        var type = activeLayer ? activeLayer.type : 'raster';

        if (controlPanel) controlPanel.style.display = type === 'control' ? 'block' : 'none';
        if (ipaPanel) ipaPanel.style.display = type === 'ipadapter' ? 'block' : 'none';
        if (regionalPanel) regionalPanel.style.display = type === 'regional' ? 'block' : 'none';
    }

    function reorderLayer(fromId, toId) {
        var fromIdx = -1, toIdx = -1;
        for (var i = 0; i < rasterLayers.length; i++) {
            if (rasterLayers[i].id === fromId) fromIdx = i;
            if (rasterLayers[i].id === toId) toIdx = i;
        }
        if (fromIdx === -1 || toIdx === -1) return;
        var moved = rasterLayers.splice(fromIdx, 1)[0];
        rasterLayers.splice(toIdx, 0, moved);
        // Reorder Konva layers to match
        rasterLayers.forEach(function(l) {
            l.konvaLayer.moveToBottom();
        });
        // Background stays at bottom, UI at top
        if (backgroundLayer) backgroundLayer.moveToBottom();
        if (uiLayer) uiLayer.moveToTop();
        stage.batchDraw();
        renderLayerList();
        History.push();
    }

    function toggleLayerVisibility(layerId) {
        for (var i = 0; i < rasterLayers.length; i++) {
            if (rasterLayers[i].id === layerId) {
                rasterLayers[i].visible = !rasterLayers[i].visible;
                rasterLayers[i].visible ? rasterLayers[i].konvaLayer.show() : rasterLayers[i].konvaLayer.hide();
                rasterLayers[i].konvaLayer.batchDraw();
                renderLayerList();
                History.push();
                break;
            }
        }
    }

    function deleteLayer(layerId) {
        if (rasterLayers.length <= 1) return; // Keep at least one
        var idx = -1;
        for (var i = 0; i < rasterLayers.length; i++) {
            if (rasterLayers[i].id === layerId) { idx = i; break; }
        }
        if (idx === -1) return;
        var info = rasterLayers[idx];
        info.konvaLayer.remove();
        rasterLayers.splice(idx, 1);
        History.push();
        if (activeLayerId === layerId) {
            activeLayerId = rasterLayers.length > 0 ? rasterLayers[rasterLayers.length - 1].id : null;
        }
        renderLayerList();
    }

    function removeLayerById(layerId, skipUndo) {
        var idx = -1;
        for (var i = 0; i < rasterLayers.length; i++) {
            if (rasterLayers[i].id === layerId) { idx = i; break; }
        }
        if (idx === -1) return;
        rasterLayers[idx].konvaLayer.remove();
        rasterLayers.splice(idx, 1);
        if (activeLayerId === layerId && rasterLayers.length > 0) {
            activeLayerId = rasterLayers[rasterLayers.length - 1].id;
        }
        renderLayerList();
    }

    // ── Tool Switching ──
    function setTool(tool) {
        if (tool === 'resetView') { resetView(); return; }
        activeTool = tool;

        document.querySelectorAll('.cv-tool-btn').forEach(function(btn) {
            btn.classList.toggle('active', btn.dataset.tool === tool);
        });

        if (els.brushSection) {
            els.brushSection.style.display = (tool === 'brush' || tool === 'eraser' || tool === 'mask') ? 'flex' : 'none';
        }

        if (brushCursor) { brushCursor.visible(false); uiLayer.batchDraw(); }

        rasterLayers.forEach(function(l) {
            l.konvaLayer.find('Image').forEach(function(img) {
                img.draggable(tool === 'select');
            });
        });

        updateCursor();
    }

    function updateCursor() {
        if (!stage) return;
        var c = stage.container();
        switch (activeTool) {
            case 'select': c.style.cursor = 'default'; break;
            case 'brush': case 'eraser': case 'mask': c.style.cursor = 'none'; break;
            case 'pan': c.style.cursor = 'grab'; break;
            default: c.style.cursor = 'default';
        }
    }

    function resetView() {
        if (!stage || !boundingBox) return;
        var container = document.getElementById('canvas-stage-container');
        if (!container) return;
        var cw = container.offsetWidth;
        var ch = container.offsetHeight;
        var bw = boundingBox.width();
        var bh = boundingBox.height();

        var padding = 100;
        var scale = Math.min((cw - padding * 2) / bw, (ch - padding * 2) / bh, 1.5);

        var bx = boundingBox.x();
        var by = boundingBox.y();
        stage.scale({ x: scale, y: scale });
        stage.position({
            x: cw / 2 - (bx + bw / 2) * scale,
            y: ch / 2 - (by + bh / 2) * scale
        });
        stage.batchDraw();
        updateVideoOverlayPosition();
    }

    // ── Keyboard ──
    function handleKeyDown(e) {
        if (localStorage.getItem('sf-active-tab') !== 'canvas') return;
        var tag = e.target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

        if ((e.ctrlKey || e.metaKey) && e.code === 'KeyZ') {
            e.preventDefault();
            if (e.shiftKey) { History.redo(); } else { History.undo(); }
            return;
        }
        if ((e.ctrlKey || e.metaKey) && e.code === 'KeyY') {
            e.preventDefault();
            History.redo();
            return;
        }

        switch (e.code) {
            case 'KeyV': setTool('select'); break;
            case 'KeyB': setTool('brush'); break;
            case 'KeyE': setTool('eraser'); break;
            case 'KeyM': setTool('mask'); break;
            case 'KeyH': setTool('pan'); break;
            case 'KeyF': resetView(); break;
            case 'Space':
                if (!isSpaceHeld) {
                    isSpaceHeld = true;
                    if (activeTool !== 'pan' && stage) stage.container().style.cursor = 'grab';
                }
                e.preventDefault();
                break;
        }
    }

    function handleKeyUp(e) {
        if (e.code === 'Space') {
            isSpaceHeld = false;
            if (!isPanning) updateCursor();
        }
    }

    // ── Right Panel ──
    function bindRightPanelEvents() {
        els.brushSizeInput.addEventListener('input', function() {
            brushSize = parseInt(this.value);
            els.brushSizeVal.textContent = brushSize;
        });

        var hardnessInput = document.getElementById('cv-brush-hardness');
        var hardnessVal = document.getElementById('cv-brush-hardness-val');
        if (hardnessInput) {
            hardnessInput.addEventListener('input', function() {
                brushHardness = parseFloat(this.value);
                if (hardnessVal) hardnessVal.textContent = brushHardness.toFixed(1);
            });
        }

        els.brushColorInput.addEventListener('input', function() {
            brushColor = this.value;
        });

        els.prompt.addEventListener('input', function() {
            genState.prompt = this.value;
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });

        els.denoise.addEventListener('input', function() {
            genState.denoise = parseFloat(this.value);
            els.denoiseVal.textContent = genState.denoise.toFixed(2);
        });

        els.steps.addEventListener('input', function() {
            genState.steps = parseInt(this.value) || 20;
            els.stepsRange.value = this.value;
        });
        els.stepsRange.addEventListener('input', function() {
            genState.steps = parseInt(this.value);
            els.steps.value = this.value;
        });

        els.cfg.addEventListener('input', function() {
            genState.cfg = parseFloat(this.value) || 7.0;
            els.cfgRange.value = this.value;
        });
        els.cfgRange.addEventListener('input', function() {
            genState.cfg = parseFloat(this.value);
            els.cfg.value = this.value;
        });

        els.model.addEventListener('change', function() {
            genState.model = this.value;
            updateTopbarModel(this.value);
            updateCanvasUIForArch(ModelUtils.detectArchFromFilename(this.value));
        });

        els.guidance.addEventListener('input', function() {
            genState.guidance = parseFloat(this.value) || 3.5;
            els.guidanceRange.value = this.value;
        });
        els.guidanceRange.addEventListener('input', function() {
            genState.guidance = parseFloat(this.value);
            els.guidance.value = this.value;
        });

        // Frames sync
        els.framesInput.addEventListener('input', function() {
            genState.frames = parseInt(this.value) || 97;
            els.framesRange.value = this.value;
            updateCanvasDurationHint();
            updateSizeLabel();
        });
        els.framesRange.addEventListener('input', function() {
            genState.frames = parseInt(this.value);
            els.framesInput.value = this.value;
            updateCanvasDurationHint();
            updateSizeLabel();
        });

        // FPS sync
        els.fpsInput.addEventListener('input', function() {
            genState.fps = parseInt(this.value) || 24;
            els.fpsRange.value = this.value;
            updateCanvasDurationHint();
        });
        els.fpsRange.addEventListener('input', function() {
            genState.fps = parseInt(this.value);
            els.fpsInput.value = this.value;
            updateCanvasDurationHint();
        });

        els.importBtn.addEventListener('click', function() { els.importFile.click(); });
        els.importFile.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                loadImageFile(this.files[0]);
                this.value = '';
            }
        });

        els.invokeBtn.addEventListener('click', function() { invoke(); });

        document.querySelectorAll('.cv-tool-btn').forEach(function(btn) {
            btn.addEventListener('click', function() { setTool(btn.dataset.tool); });
        });

        // Undo/redo buttons
        var undoBtn = document.getElementById('cv-undo');
        var redoBtn = document.getElementById('cv-redo');
        if (undoBtn) undoBtn.addEventListener('click', function() { History.undo(); });
        if (redoBtn) redoBtn.addEventListener('click', function() { History.redo(); });

        // Control layer type panel sliders — persist values to active layer object
        var controlType = document.getElementById('cv-control-type');
        if (controlType) {
            controlType.addEventListener('change', function() {
                var layer = getLayerById(activeLayerId);
                if (layer) layer.controlType = this.value;
            });
        }
        var controlWeight = document.getElementById('cv-control-weight');
        var controlWeightVal = document.getElementById('cv-control-weight-val');
        if (controlWeight) {
            controlWeight.addEventListener('input', function() {
                var val = parseFloat(this.value);
                if (controlWeightVal) controlWeightVal.textContent = val.toFixed(2);
                var layer = getLayerById(activeLayerId);
                if (layer) layer.weight = val;
            });
        }
        var controlStart = document.getElementById('cv-control-start');
        var controlStartVal = document.getElementById('cv-control-start-val');
        if (controlStart) {
            controlStart.addEventListener('input', function() {
                var val = parseFloat(this.value);
                if (controlStartVal) controlStartVal.textContent = val.toFixed(2);
                var layer = getLayerById(activeLayerId);
                if (layer) layer.startStep = val;
            });
        }
        var controlEnd = document.getElementById('cv-control-end');
        var controlEndVal = document.getElementById('cv-control-end-val');
        if (controlEnd) {
            controlEnd.addEventListener('input', function() {
                var val = parseFloat(this.value);
                if (controlEndVal) controlEndVal.textContent = val.toFixed(2);
                var layer = getLayerById(activeLayerId);
                if (layer) layer.endStep = val;
            });
        }
        // IP-Adapter weight and method — persist to layer
        var ipaWeight = document.getElementById('cv-ipa-weight');
        var ipaWeightVal = document.getElementById('cv-ipa-weight-val');
        if (ipaWeight) {
            ipaWeight.addEventListener('input', function() {
                var val = parseFloat(this.value);
                if (ipaWeightVal) ipaWeightVal.textContent = val.toFixed(2);
                var layer = getLayerById(activeLayerId);
                if (layer) layer.weight = val;
            });
        }
        var ipaMethod = document.getElementById('cv-ipa-method');
        if (ipaMethod) {
            ipaMethod.addEventListener('change', function() {
                var layer = getLayerById(activeLayerId);
                if (layer) layer.ipaMethod = this.value;
            });
        }

        // Control layer image upload
        var controlWell = document.getElementById('cv-control-well');
        var controlFile = document.getElementById('cv-control-file');
        if (controlWell && controlFile) {
            controlWell.addEventListener('click', function() { controlFile.click(); });
            controlFile.addEventListener('change', function() {
                if (this.files && this.files[0]) {
                    handleLayerImageUpload(activeLayerId, this.files[0], controlWell);
                    this.value = '';
                }
            });
            controlWell.addEventListener('dragover', function(e) { e.preventDefault(); });
            controlWell.addEventListener('drop', function(e) {
                e.preventDefault();
                var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    handleLayerImageUpload(activeLayerId, file, controlWell);
                }
            });
        }

        // IP-Adapter image upload
        var ipaWell = document.getElementById('cv-ipa-well');
        var ipaFile = document.getElementById('cv-ipa-file');
        if (ipaWell && ipaFile) {
            ipaWell.addEventListener('click', function() { ipaFile.click(); });
            ipaFile.addEventListener('change', function() {
                if (this.files && this.files[0]) {
                    handleLayerImageUpload(activeLayerId, this.files[0], ipaWell);
                    this.value = '';
                }
            });
            ipaWell.addEventListener('dragover', function(e) { e.preventDefault(); });
            ipaWell.addEventListener('drop', function(e) {
                e.preventDefault();
                var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    handleLayerImageUpload(activeLayerId, file, ipaWell);
                }
            });
        }

        // Layer add menu
        var addBtn = document.getElementById('cv-layers-add-btn');
        var typeMenu = document.getElementById('cv-layer-type-menu');
        if (addBtn && typeMenu) {
            addBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                typeMenu.style.display = typeMenu.style.display === 'none' ? 'block' : 'none';
            });
            typeMenu.addEventListener('click', function(e) {
                var item = e.target.closest('.cv-layer-type-item');
                if (!item) return;
                var type = item.dataset.type;
                var names = {
                    raster: 'Raster Layer',
                    mask: 'Inpaint Mask',
                    control: 'Control Layer',
                    ipadapter: 'IP-Adapter',
                    regional: 'Regional Prompt'
                };
                addLayer((names[type] || 'Layer') + ' ' + (rasterLayers.length + 1), type);
                typeMenu.style.display = 'none';
            });
            document.addEventListener('click', function() {
                typeMenu.style.display = 'none';
            });
        }

        // Bbox toolbar
        var bboxW = document.getElementById('cv-bbox-w');
        var bboxH = document.getElementById('cv-bbox-h');
        var bboxSnap = document.getElementById('cv-bbox-snap');
        var bboxReset = document.getElementById('cv-bbox-reset');

        if (bboxW) {
            bboxW.addEventListener('change', function() {
                var v = clampDimForArch(parseInt(this.value) || 1024);
                this.value = v;
                boundingBox.width(v);
                updateHandles();
                updateSizeLabel();
            });
        }
        if (bboxH) {
            bboxH.addEventListener('change', function() {
                var v = clampDimForArch(parseInt(this.value) || 1024);
                this.value = v;
                boundingBox.height(v);
                updateHandles();
                updateSizeLabel();
            });
        }
        if (bboxReset) {
            bboxReset.addEventListener('click', function() {
                var container = document.getElementById('canvas-stage-container');
                if (!container) return;
                var w = 1024, h = 1024;
                boundingBox.width(w);
                boundingBox.height(h);
                boundingBox.x(container.offsetWidth / 2 / stage.scaleX() - w / 2 - stage.x() / stage.scaleX());
                boundingBox.y(container.offsetHeight / 2 / stage.scaleY() - h / 2 - stage.y() / stage.scaleY());
                updateHandles();
                updateSizeLabel();
                updateBboxInputs();
                stage.batchDraw();
            });
        }

        var bboxLock = document.getElementById('cv-bbox-lock');
        if (bboxLock) {
            bboxLock.addEventListener('click', function() {
                bboxAspectLocked = !bboxAspectLocked;
                this.classList.toggle('active', bboxAspectLocked);
                if (bboxAspectLocked) {
                    bboxLockedRatio = boundingBox.width() / boundingBox.height();
                }
            });
        }

        var bboxFit = document.getElementById('cv-bbox-fit');
        if (bboxFit) {
            bboxFit.addEventListener('click', function() {
                var layer = getActiveKonvaLayer();
                if (!layer) return;
                var rect = layer.getClientRect({ skipTransform: true });
                if (rect.width < 1 || rect.height < 1) return;
                boundingBox.x(rect.x);
                boundingBox.y(rect.y);
                boundingBox.width(clampDimForArch(rect.width));
                boundingBox.height(clampDimForArch(rect.height));
                updateHandles();
                updateSizeLabel();
                stage.batchDraw();
            });
        }

        var maskFill = document.getElementById('cv-mask-fill');
        var maskClear = document.getElementById('cv-mask-clear');
        if (maskFill) {
            maskFill.addEventListener('click', function() {
                var layer = getLayerById(activeLayerId);
                if (!layer || layer.type !== 'mask') return;
                var rect = new Konva.Rect({
                    x: boundingBox.x(), y: boundingBox.y(),
                    width: boundingBox.width(), height: boundingBox.height(),
                    fill: 'rgba(239, 68, 68, 0.5)',
                    listening: false
                });
                layer.konvaLayer.add(rect);
                layer.konvaLayer.batchDraw();
                History.push();
            });
        }
        if (maskClear) {
            maskClear.addEventListener('click', function() {
                var layer = getLayerById(activeLayerId);
                if (!layer || layer.type !== 'mask') return;
                layer.konvaLayer.destroyChildren();
                layer.konvaLayer.batchDraw();
                History.push();
            });
        }
    }

    // ── Models ──
    function loadModels() {
        ModelUtils.fetchAllModels()
            .then(function(models) {
                if (!models.length) throw new Error('empty');

                els.model.innerHTML = '';
                models.forEach(function(m) {
                    var opt = document.createElement('option');
                    opt.value = m.name;
                    opt.textContent = m.name.split('/').pop();
                    els.model.appendChild(opt);
                });
                genState.model = models[0].name;
                updateTopbarModel(models[0].name);
                updateCanvasUIForArch(ModelUtils.detectArchFromFilename(models[0].name));
            })
            .catch(function() {
                els.model.innerHTML = '<option disabled selected>No models found</option>';
            });
    }

    function updateTopbarModel(modelName) {
        var badge = document.querySelector('.model-badge');
        if (!badge) return;
        // Show just the filename in the topbar badge
        var short = modelName ? modelName.split('/').pop().replace(/\.\w+$/, '') : 'No model loaded';
        badge.textContent = short;
    }

    // ── Generation ──
    function updateCanvasUIForArch(arch) {
        genState.arch = arch;
        var isFlux = arch === 'flux';
        var isVideo = arch === 'ltxv' || arch === 'wan';

        els.cfgRow.style.display = (isFlux || isVideo) ? 'none' : 'flex';
        els.guidanceRow.style.display = isFlux ? 'flex' : 'none';
        els.videoSection.style.display = isVideo ? 'block' : 'none';

        // Update invoke button label
        els.invokeBtn.textContent = isVideo ? 'Generate Video' : 'Invoke';

        // Update size label to show/hide frame count
        updateSizeLabel();

        if (isVideo) updateCanvasDurationHint();
    }

    function updateCanvasDurationHint() {
        if (!els.durationHint) return;
        var secs = (genState.frames / genState.fps).toFixed(1);
        els.durationHint.textContent = '\u2248 ' + secs + 's at ' + genState.fps + 'fps';
    }

    function getMaskLayer() {
        for (var i = 0; i < rasterLayers.length; i++) {
            if (rasterLayers[i].type === 'mask' && rasterLayers[i].visible) return rasterLayers[i];
        }
        return null;
    }

    function exportMaskAsBW() {
        var maskLayer = getMaskLayer();
        if (!maskLayer) return Promise.resolve(null);
        var children = maskLayer.konvaLayer.getChildren();
        if (children.length === 0) return Promise.resolve(null);

        return new Promise(function(resolve) {
            // Export mask layer content in bbox region
            rasterLayers.forEach(function(l) {
                if (l !== maskLayer) l.konvaLayer.hide();
            });
            uiLayer.hide();
            backgroundLayer.hide();

            var maskDataURL = stage.toDataURL({
                x: boundingBox.x(), y: boundingBox.y(),
                width: boundingBox.width(), height: boundingBox.height(),
                pixelRatio: 1
            });

            // Restore visibility
            rasterLayers.forEach(function(l) {
                if (l.visible) l.konvaLayer.show();
            });
            uiLayer.show();
            backgroundLayer.show();
            stage.batchDraw();

            // Convert to B&W: any non-transparent pixel becomes white
            var offscreen = document.createElement('canvas');
            var bw = Math.round(boundingBox.width());
            var bh = Math.round(boundingBox.height());
            offscreen.width = bw;
            offscreen.height = bh;
            var ctx = offscreen.getContext('2d');

            // Fill black (keep)
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, bw, bh);

            var img = new Image();
            img.onload = function() {
                // Draw mask content
                ctx.drawImage(img, 0, 0, bw, bh);
                // Convert any non-black pixel to white
                var imageData = ctx.getImageData(0, 0, bw, bh);
                var data = imageData.data;
                for (var i = 0; i < data.length; i += 4) {
                    if (data[i + 3] > 127) { // significant alpha = was masked
                        data[i] = 255; data[i + 1] = 255; data[i + 2] = 255; data[i + 3] = 255;
                    } else {
                        data[i] = 0; data[i + 1] = 0; data[i + 2] = 0; data[i + 3] = 255;
                    }
                }
                ctx.putImageData(imageData, 0, 0);
                resolve(offscreen.toDataURL('image/png').split(',')[1]);
            };
            img.onerror = function() { resolve(null); };
            img.src = maskDataURL;
        });
    }

    function invoke() {
        if (canvasGenerating) return;
        if (!genState.model) { showError('No model selected'); return; }
        if (!genState.prompt.trim()) { showError('Enter a prompt'); return; }

        setCanvasGenerating(true);
        var isVideo = isVideoArch();

        checkBboxContent().then(function(hasContent) {
            var seed = genState.seed === -1 ? Math.floor(Math.random() * 4294967296) : genState.seed;
            var bw = isVideo ? ModelUtils.clampVideoDimension(boundingBox.width()) : ModelUtils.clampDimension(boundingBox.width());
            var bh = isVideo ? ModelUtils.clampVideoDimension(boundingBox.height()) : ModelUtils.clampDimension(boundingBox.height());

            var maskLayerInfo = getMaskLayer();
            var hasMask = maskLayerInfo && maskLayerInfo.konvaLayer.getChildren().length > 0;

            if (!hasContent || isVideo) {
                queueWorkflow(WorkflowBuilder.build({
                    model: genState.model, prompt: genState.prompt,
                    width: bw, height: bh,
                    steps: genState.steps, cfg: genState.cfg,
                    guidance: genState.guidance, seed: seed,
                    frames: genState.frames, fps: genState.fps
                }));
            } else if (hasMask && !isVideo) {
                // Inpaint: export both init image and mask
                exportBoundingBoxRegion().then(function(initBase64) {
                    return uploadInitImage(initBase64).then(function(initName) {
                        return exportMaskAsBW().then(function(maskBase64) {
                            if (!maskBase64) {
                                // Mask export failed, fall back to img2img
                                queueWorkflow(WorkflowBuilder.buildImg2Img({
                                    model: genState.model, prompt: genState.prompt,
                                    initImageName: initName, width: bw, height: bh,
                                    steps: genState.steps, cfg: genState.cfg,
                                    guidance: genState.guidance, denoise: genState.denoise, seed: seed
                                }));
                                return;
                            }
                            return uploadInitImage(maskBase64).then(function(maskName) {
                                queueWorkflow(WorkflowBuilder.buildInpaint({
                                    model: genState.model, prompt: genState.prompt,
                                    negPrompt: '', initImageName: initName, maskImageName: maskName,
                                    width: bw, height: bh,
                                    steps: genState.steps, cfg: genState.cfg,
                                    guidance: genState.guidance, denoise: genState.denoise, seed: seed
                                }));
                            });
                        });
                    });
                }).catch(function(err) {
                    showError('Inpaint failed: ' + err.message);
                    setCanvasGenerating(false);
                });
            } else {
                exportBoundingBoxRegion().then(function(base64) {
                    return uploadInitImage(base64);
                }).then(function(imageName) {
                    queueWorkflow(WorkflowBuilder.buildImg2Img({
                        model: genState.model, prompt: genState.prompt,
                        initImageName: imageName, width: bw, height: bh,
                        steps: genState.steps, cfg: genState.cfg,
                        guidance: genState.guidance,
                        denoise: genState.denoise, seed: seed
                    }));
                }).catch(function(err) {
                    showError('Upload failed: ' + err.message);
                    setCanvasGenerating(false);
                });
            }
        });
    }

    // Check if there's actual pixel content in the bbox region by sampling pixels
    function checkBboxContent() {
        return new Promise(function(resolve) {
            uiLayer.hide();
            backgroundLayer.hide();

            var bx = boundingBox.x();
            var by = boundingBox.y();
            var bw = boundingBox.width();
            var bh = boundingBox.height();

            // Export just the raster layers in the bbox region
            var dataURL = stage.toDataURL({
                x: bx, y: by, width: bw, height: bh, pixelRatio: 0.1
            });

            uiLayer.show();
            backgroundLayer.show();
            stage.batchDraw();

            // Check if the exported region has any non-transparent pixels
            var testImg = new Image();
            testImg.onload = function() {
                var c = document.createElement('canvas');
                c.width = testImg.width;
                c.height = testImg.height;
                var ctx = c.getContext('2d');
                ctx.drawImage(testImg, 0, 0);
                var data = ctx.getImageData(0, 0, c.width, c.height).data;
                var hasPixels = false;
                for (var i = 3; i < data.length; i += 16) {
                    if (data[i] > 0) { hasPixels = true; break; }
                }
                resolve(hasPixels);
            };
            testImg.onerror = function() { resolve(false); };
            testImg.src = dataURL;
        });
    }

    function exportBoundingBoxRegion() {
        return new Promise(function(resolve) {
            uiLayer.hide();
            backgroundLayer.hide();

            var dataURL = stage.toDataURL({
                x: boundingBox.x(), y: boundingBox.y(),
                width: boundingBox.width(), height: boundingBox.height(),
                pixelRatio: 1
            });

            uiLayer.show();
            backgroundLayer.show();
            stage.batchDraw();

            resolve(dataURL.split(',')[1]);
        });
    }

    function uploadInitImage(base64Data) {
        return SerenityAPI.uploadImage(base64Data, 'canvas_init');
    }

    function collectControlLayers() {
        var controls = [];
        rasterLayers.forEach(function(l) {
            if (l.type === 'control' && l.visible && l.refImageSrc) {
                controls.push({
                    imageName: l.refImageName || null,
                    refImageSrc: l.refImageSrc,
                    controlNetModel: l.controlType ? ('control_v11p_sd15_' + l.controlType + '.safetensors') : undefined,
                    weight: l.weight || 1.0,
                    startStep: l.startStep || 0,
                    endStep: l.endStep || 1
                });
            }
        });
        return controls;
    }

    function collectIPALayers() {
        var ipas = [];
        rasterLayers.forEach(function(l) {
            if (l.type === 'ipadapter' && l.visible && l.refImageSrc) {
                ipas.push({
                    imageName: l.refImageName || null,
                    refImageSrc: l.refImageSrc,
                    weight: l.weight || 0.6,
                    ipaMethod: l.ipaMethod || 'style'
                });
            }
        });
        return ipas;
    }

    function uploadLayerImages(layers) {
        // Upload ref images that haven't been uploaded yet
        var promises = layers.map(function(l) {
            if (l.imageName) return Promise.resolve(l);
            if (!l.refImageSrc) return Promise.resolve(l);
            var base64 = l.refImageSrc.split(',')[1];
            if (!base64) return Promise.resolve(l);
            return uploadInitImage(base64).then(function(name) {
                l.imageName = name;
                return l;
            });
        });
        return Promise.all(promises);
    }

    function queueWorkflow(workflow) {
        // Apply ControlNet and IP-Adapter nodes if layers exist
        var controlLayers = collectControlLayers();
        var ipaLayers = collectIPALayers();

        var prepare = Promise.resolve();
        if (controlLayers.length > 0 || ipaLayers.length > 0) {
            prepare = Promise.all([
                uploadLayerImages(controlLayers),
                uploadLayerImages(ipaLayers)
            ]).then(function(results) {
                var controls = results[0].filter(function(l) { return l.imageName; });
                var ipas = results[1].filter(function(l) { return l.imageName; });
                if (controls.length > 0) {
                    workflow = WorkflowBuilder.applyControlNetNodes(workflow, controls);
                }
                if (ipas.length > 0) {
                    workflow = WorkflowBuilder.applyIPAdapterNodes(workflow, ipas);
                }
            });
        }

        prepare.then(function() {
            return SerenityAPI.postPrompt(workflow, {
                prompt: genState.prompt,
                model: genState.model
            });
        })
        .catch(function(err) {
            showError('Failed to queue: ' + err.message);
            setCanvasGenerating(false);
        });
    }

    function setCanvasGenerating(v) {
        canvasGenerating = v;
        var isVideo = isVideoArch();
        els.invokeBtn.disabled = v;
        if (v) {
            els.invokeBtn.textContent = 'Generating...';
        } else {
            els.invokeBtn.textContent = isVideo ? 'Generate Video' : 'Invoke';
        }
        els.invokeBtn.classList.toggle('invoking', v);
        if (v) {
            els.progress.classList.add('active');
            els.progressBar.style.width = '100%';
        } else {
            els.progress.classList.remove('active');
            els.progressBar.style.width = '0%';
            els.progressLabel.classList.remove('visible');
        }
    }

    function placeResultOnCanvas(src) {
        var img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = function() {
            var kImg = new Konva.Image({
                image: img,
                x: boundingBox.x(),
                y: boundingBox.y(),
                width: boundingBox.width(),
                height: boundingBox.height(),
                draggable: activeTool === 'select'
            });
            var layer = getActiveKonvaLayer();
            if (layer) {
                layer.add(kImg);
                layer.batchDraw();
                History.push();
            }
        };
        img.src = src;
    }

    function showError(msg) {
        if (!els.errorBanner) return;
        els.errorBanner.textContent = msg;
        els.errorBanner.classList.add('visible');
        setTimeout(function() { els.errorBanner.classList.remove('visible'); }, 5000);
    }

    // ── Video overlay on canvas ──
    function placeVideoOverlayOnCanvas(src) {
        removeVideoOverlay();
        var container = document.getElementById('canvas-stage-container');
        if (!container || !boundingBox || !stage) return;

        var overlay = document.createElement('div');
        overlay.id = 'cv-video-overlay';
        overlay.innerHTML = '<video src="' + src + '" autoplay loop muted playsinline controls' +
            ' style="width:100%;height:100%;object-fit:cover;border-radius:4px;"></video>';
        container.appendChild(overlay);
        updateVideoOverlayPosition();
    }

    function updateVideoOverlayPosition() {
        var overlay = document.getElementById('cv-video-overlay');
        if (!overlay || !boundingBox || !stage) return;
        var scale = stage.scaleX();
        var stagePos = stage.position();
        var bx = boundingBox.x() * scale + stagePos.x;
        var by = boundingBox.y() * scale + stagePos.y;
        var bw = boundingBox.width() * scale;
        var bh = boundingBox.height() * scale;
        overlay.style.cssText = 'position:absolute;left:' + bx + 'px;top:' + by + 'px;' +
            'width:' + bw + 'px;height:' + bh + 'px;pointer-events:auto;overflow:hidden;z-index:50;border-radius:4px;';
    }

    function removeVideoOverlay() {
        var overlay = document.getElementById('cv-video-overlay');
        if (overlay) overlay.remove();
    }

    // ── WebSocket ──
    function connectWS() {
        SerenityWS.on('executed', function(data) {
            if (!canvasGenerating) return;
            if (!data || !data.output) return;
            var items = data.output.images;
            var isVideoFile = false;
            if (!items && data.output.videos) {
                items = data.output.videos;
                isVideoFile = true;
            }
            if (!items || !items.length) return;

            var file = items[0];
            var src = '/view?filename=' + encodeURIComponent(file.filename) +
                '&subfolder=' + encodeURIComponent(file.subfolder || '') +
                '&type=' + encodeURIComponent(file.type || 'output');
            if (!isVideoFile) isVideoFile = /\.(webp|mp4|gif)$/i.test(file.filename);
            showCanvasPreview(src, isVideoFile);
            setCanvasGenerating(false);
        });

        SerenityWS.on('progress', function(data) {
            if (!canvasGenerating || !data) return;
            var pct = (data.value / data.max * 100).toFixed(0);
            els.progressBar.style.width = pct + '%';
            els.progressLabel.textContent = 'Step ' + data.value + ' / ' + data.max;
            els.progressLabel.classList.add('visible');
        });

        SerenityWS.on('execution_error', function(data) {
            if (!canvasGenerating) return;
            showError((data && data.exception_message) || 'Generation failed');
            setCanvasGenerating(false);
        });
    }

    // ── State Persistence ──
    function saveState() {
        if (!stage || !boundingBox) return;
        try {
            localStorage.setItem('sf-canvas-state', JSON.stringify({
                stageX: stage.x(), stageY: stage.y(), stageScale: stage.scaleX(),
                bboxX: boundingBox.x(), bboxY: boundingBox.y(),
                bboxW: boundingBox.width(), bboxH: boundingBox.height(),
                activeTool: activeTool, brushSize: brushSize
            }));
        } catch (e) { /* ignore */ }
    }

    function restoreState() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-canvas-state'));
            if (!saved) return;

            if (typeof saved.stageX === 'number') stage.position({ x: saved.stageX, y: saved.stageY });
            if (typeof saved.stageScale === 'number') stage.scale({ x: saved.stageScale, y: saved.stageScale });
            if (typeof saved.bboxX === 'number') {
                boundingBox.position({ x: saved.bboxX, y: saved.bboxY });
                boundingBox.width(clampDim(saved.bboxW || 1024));
                boundingBox.height(clampDim(saved.bboxH || 1024));
                updateHandles();
                updateSizeLabel();
            }
            if (saved.activeTool) setTool(saved.activeTool);
            if (typeof saved.brushSize === 'number') {
                brushSize = saved.brushSize;
                if (els.brushSizeInput) els.brushSizeInput.value = brushSize;
                if (els.brushSizeVal) els.brushSizeVal.textContent = brushSize;
            }
            stage.batchDraw();
        } catch (e) { /* ignore */ }
    }

    // ── Resize ──
    function resizeStage() {
        if (!stage) return;
        var container = document.getElementById('canvas-stage-container');
        if (!container) return;
        var w = container.offsetWidth;
        var h = container.offsetHeight;
        if (w < 10 || h < 10) return;
        stage.width(w);
        stage.height(h);
        stage.batchDraw();
    }

    // ── Init ──
    function init() {
        if (initialized) return;
        initialized = true;

        buildUI();
        bindRightPanelEvents();
        loadModels();
        connectWS();
        setupPreviewPanel();

        document.addEventListener('keydown', handleKeyDown);
        document.addEventListener('keyup', handleKeyUp);

        // Defer Konva init to next frame so layout is computed
        requestAnimationFrame(function() {
            initKonva();
            restoreState();

            // Check for image sent from Simple mode
            var pendingImage = localStorage.getItem('sf-send-to-canvas');
            if (pendingImage) {
                localStorage.removeItem('sf-send-to-canvas');
                try {
                    var data = JSON.parse(pendingImage);
                    if (data.src && !data.isVideo) {
                        var img = new Image();
                        img.crossOrigin = 'anonymous';
                        img.onload = function() {
                            var kImg = new Konva.Image({
                                image: img,
                                x: boundingBox.x(),
                                y: boundingBox.y(),
                                width: boundingBox.width(),
                                height: boundingBox.height(),
                                draggable: activeTool === 'select'
                            });
                            var layer = getActiveKonvaLayer();
                            if (layer) {
                                layer.add(kImg);
                                layer.batchDraw();
                            }
                        };
                        img.src = data.src;
                    }
                } catch(e) {}
            }

            resetView();
            updateCursor();
            History.push(); // Initial snapshot
            setInterval(saveState, 5000);
        });
    }

    return {
        init: init,
        resize: resizeStage,
        saveState: saveState
    };
})();
