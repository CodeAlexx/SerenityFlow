/**
 * Canvas v2 test — validates layer data model, tool registry, serialization,
 * and undo/redo logic. Runs in Node.js (no browser needed).
 *
 * We eval the compiled JS files in a mock global scope to test the logic.
 */

'use strict';
const fs = require('fs');
const path = require('path');

let passed = 0;
let failed = 0;

function assert(condition, msg) {
    if (condition) {
        passed++;
    } else {
        failed++;
        console.error('  FAIL:', msg);
    }
}

function section(name) {
    console.log('\n' + name);
}

// ── Mock Konva (minimal stubs for layer-types.ts and canvas-tools.ts) ──
global.Konva = {
    Stage: function() { return { add: function(){}, batchDraw: function(){}, scaleX: function(){ return 1; }, getPointerPosition: function(){ return {x:0,y:0}; }, toDataURL: function(){ return 'data:,'; }, x: function(){ return 0; }, y: function(){ return 0; }, width: function(){ return 800; }, height: function(){ return 600; } }; },
    Layer: function() {
        var children = [];
        return {
            add: function(c) { children.push(c); },
            batchDraw: function(){},
            destroyChildren: function() { children = []; },
            destroy: function(){},
            hide: function(){},
            show: function(){},
            opacity: function(v) { if (v !== undefined) this._op = v; return this._op || 1; },
            toDataURL: function() { return 'data:,'; },
            find: function() { return []; },
            findOne: function() { return null; },
            getChildren: function() { return children; },
            moveToBottom: function(){},
            moveToTop: function(){},
            canvas: function() { return { _canvas: { style: {} } }; },
            _op: 1,
            parent: true,
        };
    },
    Line: function(cfg) { return Object.assign({ points: function(v){ if(v) this._pts=v; return this._pts||cfg.points||[]; }, destroy: function(){}, name: function(){ return ''; }, closed: function(){}, fill: function(){} }, cfg); },
    Rect: function(cfg) { cfg=cfg||{}; var _x=cfg.x||0,_y=cfg.y||0,_w=cfg.width||0,_h=cfg.height||0; return { x:function(v){if(v!==undefined)_x=v;return _x;}, y:function(v){if(v!==undefined)_y=v;return _y;}, width:function(v){if(v!==undefined)_w=v;return _w;}, height:function(v){if(v!==undefined)_h=v;return _h;}, fill:function(){return cfg.fill||'';}, opacity:function(){return cfg.opacity||1;}, destroy:function(){}, name:function(){return cfg.name||'';}, draggable:function(){}, getClientRect:function(){return{x:_x,y:_y,width:_w,height:_h};} }; },
    Circle: function(cfg) { return Object.assign({ x: function(){}, y: function(){}, radius: function(){}, strokeWidth: function(){}, visible: function(){} }, cfg); },
    Text: function(cfg) { cfg=cfg||{}; var _t=cfg.text||''; return { x: function(v){ if(v!==undefined)this._x=v; }, y: function(){}, text: function(v){ if(v!==undefined)_t=v; return _t; }, fontSize: function(){}, fontFamily: function(){}, fontStyle: function(){}, fill: function(){}, destroy: function(){}, draggable: function(){}, name: function(){ return ''; }, moveToTop: function(){}, width: function(){ return 100; } }; },
    Image: function(cfg) { return Object.assign({ destroy: function(){}, draggable: function(){}, name: function(){ return ''; } }, cfg); },
    Shape: function() {},
};

// Mock localStorage
var _storage = {};
global.localStorage = {
    getItem: function(k) { return _storage[k] || null; },
    setItem: function(k, v) { _storage[k] = v; },
    removeItem: function(k) { delete _storage[k]; },
};

// Mock document
var _mockEls = [];
global.document = {
    body: { appendChild: function(el) { _mockEls.push(el); } },
    createElement: function(tag) {
        if (tag === 'canvas') {
            return {
                width: 0, height: 0,
                getContext: function() {
                    return {
                        fillStyle: '', fillRect: function(){},
                        drawImage: function(){},
                        getImageData: function(x,y,w,h) { return { data: new Uint8ClampedArray(w*h*4), width: w, height: h }; },
                        putImageData: function(){},
                        createLinearGradient: function() { return { addColorStop: function(){} }; },
                        globalAlpha: 1, globalCompositeOperation: 'source-over',
                    };
                },
                toDataURL: function() { return 'data:,'; },
                style: {},
            };
        }
        return { click: function(){}, href: '', download: '', style: {} };
    },
    getElementById: function() { return null; },
    querySelectorAll: function() { return []; },
    addEventListener: function() {},
};
global.window = { addEventListener: function(){}, removeEventListener: function(){}, Image: function() { this.onload = null; this.onerror = null; this.src = ''; this.crossOrigin = ''; } };
global.URL = { createObjectURL: function(){ return 'blob:'; }, revokeObjectURL: function(){} };
global.Blob = function() {};
global.Image = function() { this.onload = null; this.onerror = null; this.src = ''; };
global.FileReader = function() { this.onload = null; this.onerror = null; this.readAsText = function(){}; };
global.requestAnimationFrame = function(fn) { fn(); };
global.setTimeout = function(fn) { fn(); return 0; };
global.clearTimeout = function() {};
global.performance = { now: function(){ return Date.now(); } };
global.ResizeObserver = function() { return { observe: function(){} }; };

// ── Load compiled JS ──
var vm = require('vm');
function loadJS(filename) {
    var code = fs.readFileSync(path.join(__dirname, 'js', filename), 'utf8');
    try {
        vm.runInThisContext(code, { filename: filename });
    } catch(e) {
        console.error('Failed to load ' + filename + ':', e.message);
    }
}

// Mock fetch for SAM tests
global.fetch = function(url, opts) {
    return Promise.resolve({
        json: function() {
            return Promise.resolve({ instances: [], error: 'mock' });
        }
    });
};
global.FormData = function() { this._data = {}; this.append = function(k,v,n){ this._data[k]=v; }; };
global.atob = function(s) { return Buffer.from(s, 'base64').toString('binary'); };

loadJS('layer-types.js');
loadJS('canvas-tools.js');
loadJS('canvas-sam.js');
loadJS('canvas-video.js');
loadJS('canvas-compositor.js');
loadJS('canvas-staging.js');
loadJS('canvas-preprocess.js');
loadJS('canvas-refimages.js');
loadJS('canvas-statusbar.js');

// ══════════════════════════════════════════════════════════
//  TESTS
// ══════════════════════════════════════════════════════════

section('=== Layer Type Defaults ===');
{
    var d = LayerDefaults.draw();
    assert(d.type === 'draw', 'draw layer has type "draw"');
    assert(d.blendMode === 'Normal', 'draw layer default blend mode is Normal');
    assert(d.visible === true, 'draw layer visible by default');
    assert(d.opacity === 1, 'draw layer opacity defaults to 1');
    assert(d.locked === false, 'draw layer unlocked by default');
    assert(d.id > 0, 'draw layer gets positive id: ' + d.id);

    var m = LayerDefaults.mask();
    assert(m.type === 'mask', 'mask layer has type "mask"');
    assert(m.denoiseStrength === 0.75, 'mask denoise defaults to 0.75');
    assert(m.noiseLevel === 0, 'mask noise level defaults to 0');

    var g = LayerDefaults.guidance();
    assert(g.type === 'guidance', 'guidance layer has type "guidance"');
    assert(g.positivePrompt === '', 'guidance positive prompt empty');
    assert(g.autoNegative === false, 'guidance auto-negative off');

    var c = LayerDefaults.control();
    assert(c.type === 'control', 'control layer type correct');
    assert(c.weight === 1, 'control weight defaults to 1');
    assert(c.beginStep === 0, 'control beginStep defaults to 0');
    assert(c.endStep === 1, 'control endStep defaults to 1');

    var a = LayerDefaults.adjustment();
    assert(a.type === 'adjustment', 'adjustment layer type');
    assert(a.brightness === 0, 'adjustment brightness starts at 0');
    assert(a.contrast === 0, 'adjustment contrast starts at 0');
    assert(a.sharpness === 0, 'adjustment sharpness starts at 0');

    var t = LayerDefaults.text();
    assert(t.type === 'text', 'text layer type');
    assert(t.fontSize === 32, 'text font size defaults to 32');
    assert(t.text === 'Text', 'text defaults to "Text"');
}

section('=== Layer IDs are unique and auto-incrementing ===');
{
    var ids = [];
    for (var i = 0; i < 10; i++) {
        ids.push(LayerDefaults.draw().id);
    }
    var unique = new Set(ids);
    assert(unique.size === 10, 'all 10 layer IDs are unique');
    for (var j = 1; j < ids.length; j++) {
        assert(ids[j] > ids[j-1], 'IDs are ascending: ' + ids[j] + ' > ' + ids[j-1]);
    }
}

section('=== createByType dispatches correctly ===');
{
    var types = ['draw', 'mask', 'guidance', 'control', 'adjustment', 'text'];
    types.forEach(function(tp) {
        var layer = LayerDefaults.createByType(tp);
        assert(layer.type === tp, 'createByType("' + tp + '") → type "' + layer.type + '"');
    });
}

section('=== Validation clamps values ===');
{
    var bad = LayerDefaults.mask();
    bad.denoiseStrength = 5;
    bad.noiseLevel = -3;
    bad.opacity = 2;
    LayerValidation.sanitise(bad);
    assert(bad.denoiseStrength === 1, 'denoise clamped to 1, got: ' + bad.denoiseStrength);
    assert(bad.noiseLevel === 0, 'noise clamped to 0, got: ' + bad.noiseLevel);
    assert(bad.opacity === 1, 'opacity clamped to 1, got: ' + bad.opacity);

    var adj = LayerDefaults.adjustment();
    adj.brightness = 99;
    adj.sharpness = -5;
    LayerValidation.sanitise(adj);
    assert(adj.brightness === 1, 'brightness clamped to 1');
    assert(adj.sharpness === 0, 'sharpness clamped to 0');

    var ctrl = LayerDefaults.control();
    ctrl.weight = 10;
    ctrl.beginStep = -1;
    LayerValidation.sanitise(ctrl);
    assert(ctrl.weight === 2, 'control weight clamped to 2');
    assert(ctrl.beginStep === 0, 'control beginStep clamped to 0');

    var txt = LayerDefaults.text();
    txt.fontSize = 0;
    txt.lineHeight = 0;
    LayerValidation.sanitise(txt);
    assert(txt.fontSize === 1, 'fontSize clamped to 1');
    assert(txt.lineHeight === 0.5, 'lineHeight clamped to 0.5');
}

section('=== Validation detects errors ===');
{
    var good = LayerDefaults.draw();
    var errors = LayerValidation.validate(good);
    assert(errors.length === 0, 'valid draw layer has no errors');

    var bad2 = LayerDefaults.draw();
    bad2.name = '';
    bad2.type = 'bogus';
    var errors2 = LayerValidation.validate(bad2);
    assert(errors2.length >= 2, 'invalid layer detected: ' + errors2.join(', '));
}

section('=== NaN safety in sanitise (P1-A fix) ===');
{
    // Null/undefined inputs should get sensible defaults, not NaN
    var nullDraw = { type: 'draw', id: 1, name: 'test', visible: true, locked: false, opacity: null, position: null, blendMode: null };
    LayerValidation.sanitise(nullDraw);
    assert(!isNaN(nullDraw.opacity), 'draw opacity is not NaN after null input, got: ' + nullDraw.opacity);
    assert(nullDraw.opacity === 1, 'draw opacity defaults to 1');
    assert(nullDraw.blendMode === 'Normal', 'draw blendMode defaults to Normal (P1-B fix)');

    var nullMask = { type: 'mask', id: 2, name: 'test', visible: true, locked: false, opacity: null, position: null, fillColor: '', denoiseStrength: null, noiseLevel: undefined };
    LayerValidation.sanitise(nullMask);
    assert(!isNaN(nullMask.denoiseStrength), 'mask denoiseStrength not NaN');
    assert(nullMask.denoiseStrength === 0.75, 'mask denoiseStrength defaults to 0.75');
    assert(!isNaN(nullMask.noiseLevel), 'mask noiseLevel not NaN');
    assert(nullMask.noiseLevel === 0, 'mask noiseLevel defaults to 0');

    var nullText = { type: 'text', id: 3, name: 'test', visible: true, locked: false, opacity: null, position: null, text: '', fontFamily: '', fontSize: null, fontWeight: '', color: '', alignment: '', lineHeight: undefined };
    LayerValidation.sanitise(nullText);
    assert(!isNaN(nullText.fontSize), 'text fontSize not NaN');
    assert(nullText.fontSize === 32, 'text fontSize defaults to 32');
    assert(!isNaN(nullText.lineHeight), 'text lineHeight not NaN');
    assert(nullText.lineHeight === 1.2, 'text lineHeight defaults to 1.2');

    var nullCtrl = { type: 'control', id: 4, name: 'test', visible: true, locked: false, opacity: null, position: null, controlModel: '', weight: null, beginStep: undefined, endStep: null, controlMode: '', refImageSrc: '', refImageName: '' };
    LayerValidation.sanitise(nullCtrl);
    assert(!isNaN(nullCtrl.weight), 'control weight not NaN');
    assert(nullCtrl.weight === 1, 'control weight defaults to 1');
    assert(nullCtrl.endStep === 1, 'control endStep defaults to 1');
}

section('=== Blend Mode helpers ===');
{
    assert(BlendModeUtil.toCompositeOp('Normal') === 'source-over', 'Normal → source-over');
    assert(BlendModeUtil.toCompositeOp('Multiply') === 'multiply', 'Multiply → multiply');
    assert(BlendModeUtil.toCompositeOp('Screen') === 'screen', 'Screen → screen');
    var keys = BlendModeUtil.allKeys();
    assert(keys.length === 12, '12 blend modes, got: ' + keys.length);
    assert(BlendModeUtil.labelFor('SoftLight') === 'Soft Light', 'SoftLight label correct');
}

section('=== Serializer: toJSON/fromJSON roundtrip ===');
{
    var session = {
        version: 2,
        layers: [
            { data: LayerDefaults.draw('Test Draw'), imageData: 'data:image/png;base64,abc' },
            { data: LayerDefaults.mask('Test Mask'), imageData: 'data:,' },
        ],
        bbox: { x: 100, y: 100, width: 1024, height: 1024 },
        activeLayerId: 1,
        genSettings: { prompt: 'test', denoise: 0.75, steps: 20, cfg: 7, guidance: 3.5, seed: 42 },
    };
    var json = LayerSerializer.toJSON(session);
    assert(typeof json === 'string', 'toJSON returns string');
    assert(json.length > 100, 'JSON is non-trivial length: ' + json.length);

    var restored = LayerSerializer.fromJSON(json);
    assert(restored !== null, 'fromJSON succeeds');
    assert(restored.layers.length === 2, 'restored 2 layers');
    assert(restored.layers[0].data.type === 'draw', 'first layer is draw');
    assert(restored.layers[1].data.type === 'mask', 'second layer is mask');
    assert(restored.bbox.width === 1024, 'bbox width preserved');
    assert(restored.genSettings.prompt === 'test', 'prompt preserved');
}

section('=== Serializer: rejects bad data ===');
{
    var bad = LayerSerializer.fromJSON('not json');
    assert(bad === null, 'rejects invalid JSON');

    var bad2 = LayerSerializer.fromJSON('{"version": 99, "layers": [], "bbox": {"width": 1}}');
    assert(bad2 === null, 'rejects unsupported version');
}

section('=== Serializer: localStorage roundtrip ===');
{
    var sess = {
        version: 2,
        layers: [{ data: LayerDefaults.draw(), imageData: '' }],
        bbox: { x: 0, y: 0, width: 512, height: 512 },
        activeLayerId: null,
        genSettings: { prompt: '', denoise: 0.5, steps: 30, cfg: 5, guidance: 3, seed: -1 },
    };
    LayerSerializer.saveToLocalStorage(sess);
    var loaded = LayerSerializer.loadFromLocalStorage();
    assert(loaded !== null, 'loaded from localStorage');
    assert(loaded.layers.length === 1, 'loaded 1 layer from localStorage');
    assert(loaded.genSettings.steps === 30, 'steps preserved in localStorage');
}

section('=== Tool Registry ===');
{
    var brush = CanvasTools.get('brush');
    assert(brush !== null, 'brush tool exists');
    assert(brush.name === 'brush', 'brush tool name correct');
    assert(brush.showsBrushCursor === true, 'brush shows cursor');
    assert(typeof brush.onMouseDown === 'function', 'brush has onMouseDown');

    var eraser = CanvasTools.get('eraser');
    assert(eraser !== null, 'eraser tool exists');
    assert(eraser.cursor === 'none', 'eraser hides system cursor');

    var rect = CanvasTools.get('rect');
    assert(rect !== null, 'rect tool exists');
    assert(rect.cursor === 'crosshair', 'rect uses crosshair');

    var gradient = CanvasTools.get('gradient');
    assert(gradient !== null, 'gradient tool exists');

    var move = CanvasTools.get('move');
    assert(move !== null, 'move tool exists');
    assert(typeof move.onKeyDown === 'function', 'move tool has keyDown handler');

    var colorpicker = CanvasTools.get('colorpicker');
    assert(colorpicker !== null, 'colorpicker tool exists');

    var text = CanvasTools.get('text');
    assert(text !== null, 'text tool exists');
    assert(text.cursor === 'text', 'text tool uses text cursor');

    var fill = CanvasTools.get('fill');
    assert(fill !== null, 'fill tool exists');

    var lasso = CanvasTools.get('lasso');
    assert(lasso !== null, 'lasso tool exists');
    assert(typeof lasso.onDeactivate === 'function', 'lasso cleans up on deactivate');

    var clonestamp = CanvasTools.get('clonestamp');
    assert(clonestamp !== null, 'clonestamp tool exists');
    assert(clonestamp.showsBrushCursor === true, 'clonestamp shows brush cursor');

    var pan = CanvasTools.get('pan');
    assert(pan !== null, 'pan tool exists');
    assert(pan.cursor === 'grab', 'pan uses grab cursor');

    var select = CanvasTools.get('select');
    assert(select !== null, 'select/bbox tool exists');

    // Nonexistent tool
    var nope = CanvasTools.get('nonexistent');
    assert(nope === null, 'nonexistent tool returns null');
}

section('=== All 13 tools registered ===');
{
    var all = CanvasTools.all();
    var count = Object.keys(all).length;
    assert(count === 13, '13 tools registered, got: ' + count);
}

section('=== Fill threshold get/set ===');
{
    CanvasTools.setFillThreshold(64);
    assert(CanvasTools.getFillThreshold() === 64, 'threshold set to 64');
    CanvasTools.setFillThreshold(999);
    assert(CanvasTools.getFillThreshold() === 255, 'threshold clamped to 255');
    CanvasTools.setFillThreshold(-5);
    assert(CanvasTools.getFillThreshold() === 0, 'threshold clamped to 0');
}

section('=== Brush opacity get/set ===');
{
    CanvasTools.setBrushOpacity(0.5);
    assert(CanvasTools.getBrushOpacity() === 0.5, 'opacity set to 0.5');
    CanvasTools.setBrushOpacity(2);
    assert(CanvasTools.getBrushOpacity() === 1, 'opacity clamped to 1');
    CanvasTools.setBrushOpacity(-1);
    assert(CanvasTools.getBrushOpacity() === 0, 'opacity clamped to 0');
}

section('=== Clipboard get/set ===');
{
    assert(CanvasTools.getClipboard() === null, 'clipboard starts null');
    CanvasTools.setClipboard('data:image/png;base64,abc');
    assert(CanvasTools.getClipboard() === 'data:image/png;base64,abc', 'clipboard set');
}

section('=== Layer type labels ===');
{
    assert(LAYER_TYPE_LABELS['draw'] === 'Draw', 'draw label');
    assert(LAYER_TYPE_LABELS['mask'] === 'Mask', 'mask label');
    assert(LAYER_TYPE_LABELS['guidance'] === 'Guidance', 'guidance label');
    assert(LAYER_TYPE_LABELS['control'] === 'Control', 'control label');
    assert(LAYER_TYPE_LABELS['adjustment'] === 'Adjustment', 'adjustment label');
    assert(LAYER_TYPE_LABELS['text'] === 'Text', 'text label');
}

section('=== SAM tool registered ===');
{
    // In browser, canvas-tab.ts registers SAM at init. In tests, do it manually.
    CanvasTools.registerTool('sam', CanvasSAM.getTool());
    var samTool = CanvasTools.get('sam');
    assert(samTool !== null, 'SAM tool is registered');
    assert(samTool.name === 'sam', 'SAM tool name correct');
    assert(samTool.cursor === 'crosshair', 'SAM tool uses crosshair cursor');
    assert(typeof samTool.onActivate === 'function', 'SAM tool has onActivate');
    assert(typeof samTool.onDeactivate === 'function', 'SAM tool has onDeactivate');
    assert(typeof samTool.onMouseDown === 'function', 'SAM tool has onMouseDown');

    var allTools = CanvasTools.all();
    var toolCount = Object.keys(allTools).length;
    assert(toolCount === 14, '14 tools registered (13 + SAM), got: ' + toolCount);
}

section('=== CanvasSAM API ===');
{
    assert(typeof CanvasSAM !== 'undefined', 'CanvasSAM module loaded');
    assert(typeof CanvasSAM.getMode === 'function', 'getMode exists');
    assert(typeof CanvasSAM.setMode === 'function', 'setMode exists');
    assert(typeof CanvasSAM.isLoading === 'function', 'isLoading exists');
    assert(typeof CanvasSAM.getInstances === 'function', 'getInstances exists');
    assert(typeof CanvasSAM.getTool === 'function', 'getTool exists');

    assert(CanvasSAM.getMode() === 'text', 'default SAM mode is text');
    CanvasSAM.setMode('click');
    assert(CanvasSAM.getMode() === 'click', 'SAM mode set to click');
    CanvasSAM.setMode('exemplar');
    assert(CanvasSAM.getMode() === 'exemplar', 'SAM mode set to exemplar');
    CanvasSAM.setMode('text');

    assert(CanvasSAM.isLoading() === false, 'not loading initially');
    assert(Array.isArray(CanvasSAM.getInstances()), 'instances is an array');
    assert(CanvasSAM.getInstances().length === 0, 'no instances initially');
}

section('=== CanvasVideo API ===');
{
    assert(typeof CanvasVideo !== 'undefined', 'CanvasVideo module loaded');
    assert(typeof CanvasVideo.loadVideo === 'function', 'loadVideo exists');
    assert(typeof CanvasVideo.goToFrame === 'function', 'goToFrame exists');
    assert(typeof CanvasVideo.trackObject === 'function', 'trackObject exists');
    assert(typeof CanvasVideo.unloadVideo === 'function', 'unloadVideo exists');
    assert(typeof CanvasVideo.setMaskForCurrentFrame === 'function', 'setMaskForCurrentFrame exists');
    assert(typeof CanvasVideo.getAllMasks === 'function', 'getAllMasks exists');
    assert(typeof CanvasVideo.getMasksAsArray === 'function', 'getMasksAsArray exists');
    assert(typeof CanvasVideo.isLoaded === 'function', 'isLoaded exists');
    assert(typeof CanvasVideo.getCurrentFrame === 'function', 'getCurrentFrame exists');
    assert(typeof CanvasVideo.getTotalFrames === 'function', 'getTotalFrames exists');
    assert(typeof CanvasVideo.getFps === 'function', 'getFps exists');
    assert(typeof CanvasVideo.isTracking === 'function', 'isTracking exists');
    assert(typeof CanvasVideo.getFrameData === 'function', 'getFrameData exists');

    assert(CanvasVideo.isLoaded() === false, 'not loaded initially');
    assert(CanvasVideo.getCurrentFrame() === 0, 'current frame starts at 0');
    assert(CanvasVideo.getTotalFrames() === 0, 'no frames initially');
    assert(CanvasVideo.getFps() === 24, 'default fps is 24');
    assert(CanvasVideo.isTracking() === false, 'not tracking initially');
    assert(CanvasVideo.getFrameData(0) === null, 'no frame data at index 0');

    var masks = CanvasVideo.getAllMasks();
    assert(masks instanceof Map, 'masks is a Map');
    assert(masks.size === 0, 'no masks initially');

    var masksArr = CanvasVideo.getMasksAsArray();
    assert(Array.isArray(masksArr), 'getMasksAsArray returns array');
    assert(masksArr.length === 0, 'no masks in array initially');
}

section('=== Compositor API ===');
{
    assert(typeof Compositor !== 'undefined', 'Compositor module loaded');
    assert(typeof Compositor.detectMode === 'function', 'detectMode exists');
    assert(typeof Compositor.compose === 'function', 'compose exists');
    assert(typeof Compositor.composeBatch === 'function', 'composeBatch exists');
    assert(typeof Compositor.flattenDrawLayers === 'function', 'flattenDrawLayers exists');
    assert(typeof Compositor.flattenMaskLayers === 'function', 'flattenMaskLayers exists');
    assert(typeof Compositor.collectGuidanceRegions === 'function', 'collectGuidanceRegions exists');
    assert(typeof Compositor.collectControlInputs === 'function', 'collectControlInputs exists');
    assert(typeof Compositor.applyAdjustments === 'function', 'applyAdjustments exists');
}

// Mock CustomEvent for staging tests
global.CustomEvent = function(type) { this.type = type; };

// Shared compositor mock helper (must be outside block scope)
function mockCtx(layers) {
    return {
        stage: new Konva.Stage(),
        boundingBox: new Konva.Rect({ x: 0, y: 0, width: 512, height: 512 }),
        backgroundLayer: new Konva.Layer(),
        uiLayer: new Konva.Layer(),
        canvasLayers: layers,
        genState: { model: 'test', prompt: 'test', denoise: 0.75, steps: 20, cfg: 7, guidance: 3.5, seed: -1, arch: 'sd15', frames: 97, fps: 24 },
        uploadImage: function(b64) { return Promise.resolve('img_' + Date.now()); },
    };
}

section('=== Compositor Mode Detection ===');
{

    // Empty canvas → txt2img
    var emptyCtx = mockCtx([]);
    assert(Compositor.detectMode(emptyCtx) === 'txt2img', 'empty canvas → txt2img');

    // Draw layer with content filling the bbox → img2img
    var drawLayer = new Konva.Layer();
    drawLayer.add(new Konva.Rect({ x: 0, y: 0, width: 512, height: 512 }));
    var drawData = LayerDefaults.draw();
    var imgCtx = mockCtx([{ data: drawData, konvaLayer: drawLayer }]);
    assert(Compositor.detectMode(imgCtx) === 'img2img', 'draw content only → img2img');

    // Draw + mask → inpaint (draw content covers bbox so not outpaint)
    var maskLayer = new Konva.Layer();
    maskLayer.add(new Konva.Rect({ width: 50, height: 50 }));
    var maskData = LayerDefaults.mask();
    var inpaintCtx = mockCtx([
        { data: drawData, konvaLayer: drawLayer },
        { data: maskData, konvaLayer: maskLayer },
    ]);
    assert(Compositor.detectMode(inpaintCtx) === 'inpaint', 'draw + mask → inpaint');

    // Guidance layer → regional
    var guidLayer = new Konva.Layer();
    guidLayer.add(new Konva.Rect({ width: 50, height: 50 }));
    var guidData = LayerDefaults.guidance();
    guidData.positivePrompt = 'a red hat';
    var regionalCtx = mockCtx([
        { data: drawData, konvaLayer: drawLayer },
        { data: guidData, konvaLayer: guidLayer },
    ]);
    assert(Compositor.detectMode(regionalCtx) === 'regional', 'guidance with prompt → regional');

    // Empty guidance (no prompt) should NOT trigger regional
    var emptyGuidData = LayerDefaults.guidance();
    var emptyGuidLayer = new Konva.Layer();
    emptyGuidLayer.add(new Konva.Rect({ width: 50, height: 50 }));
    var noRegionalCtx = mockCtx([
        { data: drawData, konvaLayer: drawLayer },
        { data: emptyGuidData, konvaLayer: emptyGuidLayer },
    ]);
    assert(Compositor.detectMode(noRegionalCtx) === 'img2img', 'guidance without prompt → img2img (not regional)');

    // Hidden layers should not affect mode
    var hiddenMask = LayerDefaults.mask();
    hiddenMask.visible = false;
    var hiddenMaskLayer = new Konva.Layer();
    hiddenMaskLayer.add(new Konva.Rect({ width: 50, height: 50 }));
    var hiddenCtx = mockCtx([
        { data: drawData, konvaLayer: drawLayer },
        { data: hiddenMask, konvaLayer: hiddenMaskLayer },
    ]);
    assert(Compositor.detectMode(hiddenCtx) === 'img2img', 'hidden mask → img2img (not inpaint)');
}

section('=== Compositor Batch ===');
{
    var batchCtx = mockCtx([]);
    Compositor.composeBatch(batchCtx, 4).then(function(payloads) {
        assert(payloads.length === 4, 'batch produces 4 payloads');
        // All should be txt2img
        assert(payloads[0].mode === 'txt2img', 'batch[0] mode is txt2img');
        // Seeds should differ (except possibly batch[0])
        var seeds = payloads.map(function(p) { return p.params.seed; });
        var uniqueSeeds = new Set(seeds);
        assert(uniqueSeeds.size >= 3, 'batch seeds are mostly unique: ' + uniqueSeeds.size + '/4');
    });
}

section('=== Staging API ===');
{
    assert(typeof CanvasStaging !== 'undefined', 'CanvasStaging module loaded');
    assert(typeof CanvasStaging.activate === 'function', 'activate exists');
    assert(typeof CanvasStaging.deactivate === 'function', 'deactivate exists');
    assert(typeof CanvasStaging.accept === 'function', 'accept exists');
    assert(typeof CanvasStaging.reject === 'function', 'reject exists');
    assert(typeof CanvasStaging.regenerate === 'function', 'regenerate exists');
    assert(typeof CanvasStaging.nextResult === 'function', 'nextResult exists');
    assert(typeof CanvasStaging.prevResult === 'function', 'prevResult exists');
    assert(typeof CanvasStaging.toggleCompare === 'function', 'toggleCompare exists');
    assert(typeof CanvasStaging.togglePartialMask === 'function', 'togglePartialMask exists');
    assert(typeof CanvasStaging.isActive === 'function', 'isActive exists');
    assert(typeof CanvasStaging.isCompareMode === 'function', 'isCompareMode exists');
    assert(typeof CanvasStaging.isPartialMaskMode === 'function', 'isPartialMaskMode exists');

    assert(CanvasStaging.isActive() === false, 'not active initially');
    assert(CanvasStaging.getCurrentIndex() === 0, 'currentIndex starts at 0');
    assert(CanvasStaging.getResults().length === 0, 'no results initially');
    assert(CanvasStaging.isCompareMode() === false, 'compare mode off initially');
    assert(CanvasStaging.isPartialMaskMode() === false, 'partial mask off initially');
}

section('=== Staging lifecycle ===');
{
    // Build a mock tool context
    var stgStage = new Konva.Stage();
    var stgUiLayer = new Konva.Layer();
    stgStage.add(stgUiLayer);
    var stgBbox = new Konva.Rect({ x: 100, y: 100, width: 512, height: 512 });
    var stgCtx = {
        stage: stgStage,
        uiLayer: stgUiLayer,
        brushCursor: new Konva.Circle({}),
        boundingBox: stgBbox,
        getActiveLayer: function() { return null; },
        getActiveKonvaLayer: function() { return new Konva.Layer(); },
        getRelativePointerPosition: function() { return { x: 0, y: 0 }; },
        pushHistory: function() {},
        pushHistoryGrouped: function() {},
        setActiveTool: function() {},
        getBrushSize: function() { return 20; },
        setBrushSize: function() {},
        getBrushColor: function() { return '#fff'; },
        setBrushColor: function() {},
        getBrushHardness: function() { return 1; },
        setBrushHardness: function() {},
        getBrushOpacity: function() { return 1; },
        setBrushOpacity: function() {},
        addLayer: function() { return { data: LayerDefaults.draw(), konvaLayer: new Konva.Layer() }; },
        deleteActiveLayer: function() {},
        flattenVisible: function() {},
        duplicateActiveLayer: function() {},
    };

    // Activate with 3 results
    CanvasStaging.activate([
        { src: 'data:image/png;base64,a', isVideo: false, seed: 1, index: 0 },
        { src: 'data:image/png;base64,b', isVideo: false, seed: 2, index: 1 },
        { src: 'data:image/png;base64,c', isVideo: false, seed: 3, index: 2 },
    ], stgCtx);

    assert(CanvasStaging.isActive() === true, 'staging is active after activate');
    assert(CanvasStaging.getResults().length === 3, '3 results staged');
    assert(CanvasStaging.getCurrentIndex() === 0, 'starts at index 0');

    // Navigate
    CanvasStaging.nextResult();
    assert(CanvasStaging.getCurrentIndex() === 1, 'next → index 1');
    CanvasStaging.nextResult();
    assert(CanvasStaging.getCurrentIndex() === 2, 'next → index 2');
    CanvasStaging.nextResult();
    assert(CanvasStaging.getCurrentIndex() === 0, 'next wraps → index 0');
    CanvasStaging.prevResult();
    assert(CanvasStaging.getCurrentIndex() === 2, 'prev wraps → index 2');

    // Compare mode
    CanvasStaging.toggleCompare();
    assert(CanvasStaging.isCompareMode() === true, 'compare mode on');
    CanvasStaging.toggleCompare();
    assert(CanvasStaging.isCompareMode() === false, 'compare mode off');

    // Partial mask mode
    CanvasStaging.togglePartialMask();
    assert(CanvasStaging.isPartialMaskMode() === true, 'partial mask on');
    CanvasStaging.togglePartialMask();
    assert(CanvasStaging.isPartialMaskMode() === false, 'partial mask off');

    // Reject clears staging
    CanvasStaging.reject();
    assert(CanvasStaging.isActive() === false, 'staging deactivated after reject');
    assert(CanvasStaging.getResults().length === 0, 'results cleared after reject');
}

section('=== Preprocessor API ===');
{
    assert(typeof CanvasPreprocess !== 'undefined', 'CanvasPreprocess module loaded');
    assert(typeof CanvasPreprocess.getAllMethods === 'function', 'getAllMethods exists');
    assert(typeof CanvasPreprocess.isProcessing === 'function', 'isProcessing exists');
    assert(typeof CanvasPreprocess.process === 'function', 'process exists');
    assert(typeof CanvasPreprocess.processActiveControlLayer === 'function', 'processActiveControlLayer exists');

    assert(CanvasPreprocess.isProcessing() === false, 'not processing initially');

    var methods = CanvasPreprocess.getAllMethods();
    assert(typeof methods === 'object', 'methods is an object');
    var methodKeys = Object.keys(methods);
    assert(methodKeys.length === 9, '9 preprocessor methods, got: ' + methodKeys.length);

    // Verify all expected methods
    var expected = ['canny', 'depth', 'lineart', 'pose', 'soft_edge', 'tile', 'normal', 'color', 'scribble'];
    expected.forEach(function(m) {
        assert(methods[m] !== undefined, 'method "' + m + '" exists');
    });

    // process() with unknown method should reject
    CanvasPreprocess.process('nonexistent', 'data:image/png;base64,abc')
        .then(function() { assert(false, 'should have rejected unknown method'); })
        .catch(function(err) { assert(err.message.indexOf('Unknown') >= 0, 'rejects unknown method: ' + err.message); });
}

section('=== Reference Images ===');
{
    assert(typeof CanvasRefImages !== 'undefined', 'CanvasRefImages loaded');
    assert(CanvasRefImages.getAll().length === 0, 'no refs initially');

    var ref1 = CanvasRefImages.add('data:image/png;base64,test1');
    assert(ref1.id.startsWith('ref_'), 'ref has id');
    assert(ref1.weight === 1.0, 'default weight 1.0');
    assert(ref1.method === 'style', 'default method style');
    assert(CanvasRefImages.getAll().length === 1, '1 ref after add');

    var ref2 = CanvasRefImages.add('data:image/png;base64,test2');
    assert(CanvasRefImages.getAll().length === 2, '2 refs after second add');

    var payload = CanvasRefImages.getForPayload();
    assert(payload.length === 2, 'payload has 2 entries');
    assert(payload[0].method === 'style', 'payload method correct');

    CanvasRefImages.remove(ref1.id);
    assert(CanvasRefImages.getAll().length === 1, '1 ref after remove');
    CanvasRefImages.remove(ref2.id);
    assert(CanvasRefImages.getAll().length === 0, '0 refs after remove all');
}

section('=== Context Menu ===');
{
    assert(typeof CanvasContextMenu !== 'undefined', 'CanvasContextMenu loaded');
    assert(typeof CanvasContextMenu.show === 'function', 'show exists');
    assert(typeof CanvasContextMenu.hide === 'function', 'hide exists');
}

section('=== Smart Crop ===');
{
    assert(typeof SmartCrop !== 'undefined', 'SmartCrop loaded');
    assert(typeof SmartCrop.checkBboxCutsObjects === 'function', 'checkBboxCutsObjects exists');
    assert(typeof SmartCrop.snapToObjects === 'function', 'snapToObjects exists');

    // With no SAM instances, should not cut
    var cropCtx = mockCtx([]);
    var result = SmartCrop.checkBboxCutsObjects(cropCtx);
    assert(result.cuts === false, 'no SAM instances → no cut');
}

section('=== Zoom Controls ===');
{
    assert(typeof CanvasZoom !== 'undefined', 'CanvasZoom loaded');
    assert(typeof CanvasZoom.fitToScreen === 'function', 'fitToScreen exists');
    assert(typeof CanvasZoom.zoomTo100 === 'function', 'zoomTo100 exists');
    assert(typeof CanvasZoom.getZoomPercent === 'function', 'getZoomPercent exists');

    var zoomCtx = mockCtx([]);
    assert(CanvasZoom.getZoomPercent(zoomCtx) === 100, 'default zoom is 100%');
}

section('=== Layer Thumbnails ===');
{
    assert(typeof LayerThumbnails !== 'undefined', 'LayerThumbnails loaded');
    assert(LayerThumbnails.getSize() === 36, 'thumbnail size is 36px');
}

section('=== Layer Solo Mode ===');
{
    assert(typeof LayerSolo !== 'undefined', 'LayerSolo loaded');
    assert(LayerSolo.isActive() === false, 'solo not active initially');

    // Create mock layers
    var soloLayer1 = new Konva.Layer();
    var soloLayer2 = new Konva.Layer();
    var soloData1 = LayerDefaults.draw('Layer 1');
    var soloData2 = LayerDefaults.draw('Layer 2');
    var soloLayers = [
        { data: soloData1, konvaLayer: soloLayer1 },
        { data: soloData2, konvaLayer: soloLayer2 },
    ];

    // Solo layer 1
    LayerSolo.toggle(soloData1.id, soloLayers);
    assert(LayerSolo.isActive() === true, 'solo is active');
    assert(LayerSolo.isSoloed(soloData1.id) === true, 'layer 1 is soloed');
    assert(soloLayers[0].data.visible === true, 'soloed layer visible');
    assert(soloLayers[1].data.visible === false, 'other layer hidden');

    // Un-solo
    LayerSolo.toggle(soloData1.id, soloLayers);
    assert(LayerSolo.isActive() === false, 'solo deactivated');
    assert(soloLayers[0].data.visible === true, 'layer 1 restored');
    assert(soloLayers[1].data.visible === true, 'layer 2 restored');
}

section('=== Status Bar ===');
{
    assert(typeof CanvasStatusBar !== 'undefined', 'CanvasStatusBar loaded');
    assert(typeof CanvasStatusBar.create === 'function', 'create exists');
    assert(typeof CanvasStatusBar.updateCursor === 'function', 'updateCursor exists');
    assert(typeof CanvasStatusBar.updateZoom === 'function', 'updateZoom exists');
    assert(typeof CanvasStatusBar.updateActiveLayer === 'function', 'updateActiveLayer exists');
    assert(typeof CanvasStatusBar.updateGenStatus === 'function', 'updateGenStatus exists');
    assert(typeof CanvasStatusBar.updateVram === 'function', 'updateVram exists');
    assert(typeof CanvasStatusBar.updateModel === 'function', 'updateModel exists');

    assert(CanvasStatusBar.getGenStatus() === 'idle', 'initial gen status is idle');
    assert(CanvasStatusBar.getVramMb() === 0, 'initial VRAM is 0');
    assert(CanvasStatusBar.getModelName() === '', 'initial model is empty');

    // Update gen status
    CanvasStatusBar.updateGenStatus('generating');
    assert(CanvasStatusBar.getGenStatus() === 'generating', 'gen status updated to generating');
    CanvasStatusBar.updateGenStatus('complete');
    assert(CanvasStatusBar.getGenStatus() === 'complete', 'gen status updated to complete');
    CanvasStatusBar.updateGenStatus('idle');

    // Update VRAM
    CanvasStatusBar.updateVram(8192);
    assert(CanvasStatusBar.getVramMb() === 8192, 'VRAM set to 8192');

    // Update model
    CanvasStatusBar.updateModel('models/flux-dev.safetensors');
    assert(CanvasStatusBar.getModelName() === 'models/flux-dev.safetensors', 'model name set');
}

// ══════════════════════════════════════════════════════════
//  SUMMARY
// ══════════════════════════════════════════════════════════
console.log('\n' + '═'.repeat(50));
console.log('Passed: ' + passed + '  Failed: ' + failed);
if (failed > 0) {
    console.log('SOME TESTS FAILED');
    process.exit(1);
} else {
    console.log('ALL TESTS PASSED');
}
