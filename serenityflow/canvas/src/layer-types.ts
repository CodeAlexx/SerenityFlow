/**
 * Layer System Data Model — SerenityFlow Canvas v2
 *
 * Typed layer definitions, blend modes, drawing primitives, validation,
 * serialization. All types are global (no modules) to match SF conventions.
 */

// ── Blend Modes ──

var BlendMode = {
    Normal:     'source-over',
    Multiply:   'multiply',
    Screen:     'screen',
    Overlay:    'overlay',
    SoftLight:  'soft-light',
    HardLight:  'hard-light',
    ColorDodge: 'color-dodge',
    ColorBurn:  'color-burn',
    Darken:     'darken',
    Lighten:    'lighten',
    Difference: 'difference',
    Exclusion:  'exclusion',
} as const;

type BlendModeKey = keyof typeof BlendMode;
type BlendModeValue = typeof BlendMode[BlendModeKey];

var BLEND_MODE_LABELS: Record<BlendModeKey, string> = {
    Normal:     'Normal',
    Multiply:   'Multiply',
    Screen:     'Screen',
    Overlay:    'Overlay',
    SoftLight:  'Soft Light',
    HardLight:  'Hard Light',
    ColorDodge: 'Color Dodge',
    ColorBurn:  'Color Burn',
    Darken:     'Darken',
    Lighten:    'Lighten',
    Difference: 'Difference',
    Exclusion:  'Exclusion',
};

// ── Layer Type Enum ──

var LayerType = {
    Draw:       'draw',
    Mask:       'mask',
    Guidance:   'guidance',
    Control:    'control',
    Adjustment: 'adjustment',
    Text:       'text',
} as const;

type LayerTypeValue = typeof LayerType[keyof typeof LayerType];

var LAYER_TYPE_LABELS: Record<LayerTypeValue, string> = {
    draw:       'Draw',
    mask:       'Mask',
    guidance:   'Guidance',
    control:    'Control',
    adjustment: 'Adjustment',
    text:       'Text',
};

// ── Drawing Primitives ──

interface StrokePrimitive {
    kind: 'stroke';
    points: number[];
    color: string;
    width: number;
    opacity: number;
    compositeOp: string;  // 'source-over' | 'destination-out'
    lineCap: string;
    lineJoin: string;
}

interface RectPrimitive {
    kind: 'rect';
    x: number;
    y: number;
    width: number;
    height: number;
    fill: string;
    opacity: number;
}

interface GradientPrimitive {
    kind: 'gradient';
    x0: number; y0: number;
    x1: number; y1: number;
    colorStops: Array<{ offset: number; color: string }>;
    type: 'linear' | 'radial';
    radius?: number;
}

interface PastedImagePrimitive {
    kind: 'image';
    src: string;       // data URL or blob URL
    x: number;
    y: number;
    width: number;
    height: number;
    opacity: number;
}

type DrawingPrimitive = StrokePrimitive | RectPrimitive | GradientPrimitive | PastedImagePrimitive;

// ── Base Layer Properties ──

interface LayerBase {
    id: number;
    name: string;
    type: LayerTypeValue;
    visible: boolean;
    locked: boolean;
    opacity: number;  // 0..1
    position: { x: number; y: number };
}

// ── Concrete Layer Types ──

interface DrawLayerData extends LayerBase {
    type: 'draw';
    blendMode: BlendModeKey;
}

interface MaskLayerData extends LayerBase {
    type: 'mask';
    fillColor: string;
    denoiseStrength: number;  // 0..1
    noiseLevel: number;       // 0..1
}

interface GuidanceLayerData extends LayerBase {
    type: 'guidance';
    positivePrompt: string;
    negativePrompt: string;
    referenceImages: string[];  // data URLs
    autoNegative: boolean;
}

interface ControlLayerData extends LayerBase {
    type: 'control';
    controlModel: string;
    weight: number;           // 0..2
    beginStep: number;        // 0..1
    endStep: number;          // 0..1
    controlMode: string;
    refImageSrc: string;
    refImageName: string;
}

interface AdjustmentLayerData extends LayerBase {
    type: 'adjustment';
    brightness: number;    // -1..1
    contrast: number;      // -1..1
    saturation: number;    // -1..1
    temperature: number;   // -1..1
    tint: number;          // -1..1
    sharpness: number;     // 0..1
    hasMask: boolean;
}

interface TextLayerData extends LayerBase {
    type: 'text';
    text: string;
    fontFamily: string;
    fontSize: number;
    fontWeight: string;
    color: string;
    alignment: string;
    lineHeight: number;
}

type LayerData = DrawLayerData | MaskLayerData | GuidanceLayerData
              | ControlLayerData | AdjustmentLayerData | TextLayerData;

// ── Runtime Layer (data + Konva reference) ──

interface CanvasLayer {
    data: LayerData;
    konvaLayer: Konva.Layer;
}

// ── Canvas Session State ──

interface CanvasSessionState {
    version: number;
    layers: LayerSerialised[];
    bbox: { x: number; y: number; width: number; height: number };
    activeLayerId: number | null;
    genSettings: {
        prompt: string;
        denoise: number;
        steps: number;
        cfg: number;
        guidance: number;
        seed: number;
    };
}

interface LayerSerialised {
    data: LayerData;
    imageData: string;  // base64 PNG of layer content
}

// ── Snapshot for Undo/Redo ──

interface CanvasSnapshot {
    layers: LayerSerialised[];
    bbox: { x: number; y: number; width: number; height: number };
    activeLayerId: number | null;
}

// ── Default Factories ──

var LayerDefaults = (function() {
    'use strict';

    var _nextId = 0;

    function nextId(): number {
        return ++_nextId;
    }

    function setIdCounter(n: number): void {
        _nextId = n;
    }

    function getIdCounter(): number {
        return _nextId;
    }

    function base(name: string, type: LayerTypeValue): LayerBase {
        return {
            id: nextId(),
            name: name,
            type: type,
            visible: true,
            locked: false,
            opacity: 1,
            position: { x: 0, y: 0 },
        };
    }

    function draw(name?: string): DrawLayerData {
        return {
            ...base(name || 'Draw Layer', LayerType.Draw),
            type: 'draw',
            blendMode: 'Normal',
        };
    }

    function mask(name?: string): MaskLayerData {
        return {
            ...base(name || 'Mask', LayerType.Mask),
            type: 'mask',
            fillColor: 'rgba(239, 68, 68, 0.5)',
            denoiseStrength: 0.75,
            noiseLevel: 0,
        };
    }

    function guidance(name?: string): GuidanceLayerData {
        return {
            ...base(name || 'Guidance', LayerType.Guidance),
            type: 'guidance',
            positivePrompt: '',
            negativePrompt: '',
            referenceImages: [],
            autoNegative: false,
        };
    }

    function control(name?: string): ControlLayerData {
        return {
            ...base(name || 'Control', LayerType.Control),
            type: 'control',
            controlModel: '',
            weight: 1,
            beginStep: 0,
            endStep: 1,
            controlMode: 'balanced',
            refImageSrc: '',
            refImageName: '',
        };
    }

    function adjustment(name?: string): AdjustmentLayerData {
        return {
            ...base(name || 'Adjustment', LayerType.Adjustment),
            type: 'adjustment',
            brightness: 0,
            contrast: 0,
            saturation: 0,
            temperature: 0,
            tint: 0,
            sharpness: 0,
            hasMask: false,
        };
    }

    function text(name?: string): TextLayerData {
        return {
            ...base(name || 'Text', LayerType.Text),
            type: 'text',
            text: 'Text',
            fontFamily: 'Inter, system-ui, sans-serif',
            fontSize: 32,
            fontWeight: 'normal',
            color: '#ffffff',
            alignment: 'left',
            lineHeight: 1.2,
        };
    }

    function createByType(type: LayerTypeValue, name?: string): LayerData {
        switch (type) {
            case 'draw':       return draw(name);
            case 'mask':       return mask(name);
            case 'guidance':   return guidance(name);
            case 'control':    return control(name);
            case 'adjustment': return adjustment(name);
            case 'text':       return text(name);
            default:           return draw(name);
        }
    }

    return {
        nextId: nextId,
        setIdCounter: setIdCounter,
        getIdCounter: getIdCounter,
        draw: draw,
        mask: mask,
        guidance: guidance,
        control: control,
        adjustment: adjustment,
        text: text,
        createByType: createByType,
    };
})();

// ── Validation ──

var LayerValidation = (function() {
    'use strict';

    function clamp(val: number, min: number, max: number): number {
        return Math.max(min, Math.min(max, val));
    }

    function validateBase(data: LayerBase): string[] {
        var errors: string[] = [];
        if (typeof data.id !== 'number' || data.id < 0) errors.push('id must be a non-negative number');
        if (typeof data.name !== 'string' || data.name.length === 0) errors.push('name must be non-empty');
        if (typeof data.opacity !== 'number') errors.push('opacity must be a number');
        if (typeof data.visible !== 'boolean') errors.push('visible must be boolean');
        if (typeof data.locked !== 'boolean') errors.push('locked must be boolean');
        return errors;
    }

    function sanitiseBase(data: LayerBase): void {
        data.opacity = clamp(data.opacity, 0, 1);
        if (!data.name) data.name = 'Untitled';
        if (!data.position) data.position = { x: 0, y: 0 };
    }

    function sanitise(data: LayerData): LayerData {
        sanitiseBase(data);
        switch (data.type) {
            case 'mask':
                data.denoiseStrength = clamp(data.denoiseStrength, 0, 1);
                data.noiseLevel = clamp(data.noiseLevel, 0, 1);
                break;
            case 'control':
                data.weight = clamp(data.weight, 0, 2);
                data.beginStep = clamp(data.beginStep, 0, 1);
                data.endStep = clamp(data.endStep, 0, 1);
                break;
            case 'adjustment':
                data.brightness = clamp(data.brightness, -1, 1);
                data.contrast = clamp(data.contrast, -1, 1);
                data.saturation = clamp(data.saturation, -1, 1);
                data.temperature = clamp(data.temperature, -1, 1);
                data.tint = clamp(data.tint, -1, 1);
                data.sharpness = clamp(data.sharpness, 0, 1);
                break;
            case 'text':
                data.fontSize = clamp(data.fontSize, 1, 1000);
                data.lineHeight = clamp(data.lineHeight, 0.5, 5);
                break;
        }
        return data;
    }

    function validate(data: LayerData): string[] {
        var errors = validateBase(data);
        var validTypes = ['draw', 'mask', 'guidance', 'control', 'adjustment', 'text'];
        if (validTypes.indexOf(data.type) === -1) {
            errors.push('invalid layer type: ' + data.type);
        }
        return errors;
    }

    function validateSession(state: CanvasSessionState): string[] {
        var errors: string[] = [];
        if (state.version !== 1) errors.push('unsupported session version: ' + state.version);
        if (!Array.isArray(state.layers)) errors.push('layers must be an array');
        if (!state.bbox || typeof state.bbox.width !== 'number') errors.push('bbox is invalid');
        if (state.layers) {
            state.layers.forEach(function(l, i) {
                var layerErrors = validate(l.data);
                layerErrors.forEach(function(e) {
                    errors.push('layer[' + i + ']: ' + e);
                });
            });
        }
        return errors;
    }

    return {
        clamp: clamp,
        validate: validate,
        sanitise: sanitise,
        validateSession: validateSession,
    };
})();

// ── Serialization ──

var LayerSerializer = (function() {
    'use strict';

    var SESSION_VERSION = 1;
    var STORAGE_KEY = 'sf-canvas-session';

    function buildSessionState(
        layers: CanvasLayer[],
        bbox: { x: number; y: number; width: number; height: number },
        activeLayerId: number | null,
        genSettings: CanvasSessionState['genSettings']
    ): Promise<CanvasSessionState> {
        // Capture each layer's pixel content as base64 PNG
        var serialised: LayerSerialised[] = [];
        var promises: Promise<void>[] = [];

        layers.forEach(function(layer) {
            var entry: LayerSerialised = {
                data: JSON.parse(JSON.stringify(layer.data)),
                imageData: '',
            };
            serialised.push(entry);
            promises.push(new Promise<void>(function(resolve) {
                try {
                    entry.imageData = layer.konvaLayer.toDataURL();
                } catch (e) {
                    entry.imageData = '';
                }
                resolve();
            }));
        });

        return Promise.all(promises).then(function() {
            return {
                version: SESSION_VERSION,
                layers: serialised,
                bbox: bbox,
                activeLayerId: activeLayerId,
                genSettings: genSettings,
            };
        });
    }

    function toJSON(state: CanvasSessionState): string {
        return JSON.stringify(state);
    }

    function fromJSON(json: string): CanvasSessionState | null {
        try {
            var parsed = JSON.parse(json) as CanvasSessionState;
            var errors = LayerValidation.validateSession(parsed);
            if (errors.length > 0) {
                console.warn('Canvas session validation errors:', errors);
                return null;
            }
            // Sanitise all layers
            parsed.layers.forEach(function(l) {
                LayerValidation.sanitise(l.data);
            });
            return parsed;
        } catch (e) {
            console.error('Failed to parse canvas session:', e);
            return null;
        }
    }

    function saveToLocalStorage(state: CanvasSessionState): void {
        try {
            localStorage.setItem(STORAGE_KEY, toJSON(state));
        } catch (e) {
            console.warn('Failed to save canvas session to localStorage:', e);
        }
    }

    function loadFromLocalStorage(): CanvasSessionState | null {
        var raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return null;
        return fromJSON(raw);
    }

    function downloadAsFile(state: CanvasSessionState, filename?: string): void {
        var json = toJSON(state);
        var blob = new Blob([json], { type: 'application/json' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = filename || ('canvas_' + Date.now() + '.serenity-canvas');
        a.click();
        URL.revokeObjectURL(url);
    }

    function loadFromFile(file: File): Promise<CanvasSessionState | null> {
        return new Promise(function(resolve) {
            var reader = new FileReader();
            reader.onload = function(ev) {
                var text = (ev.target as FileReader).result as string;
                resolve(fromJSON(text));
            };
            reader.onerror = function() { resolve(null); };
            reader.readAsText(file);
        });
    }

    return {
        SESSION_VERSION: SESSION_VERSION,
        STORAGE_KEY: STORAGE_KEY,
        buildSessionState: buildSessionState,
        toJSON: toJSON,
        fromJSON: fromJSON,
        saveToLocalStorage: saveToLocalStorage,
        loadFromLocalStorage: loadFromLocalStorage,
        downloadAsFile: downloadAsFile,
        loadFromFile: loadFromFile,
    };
})();

// ── Blend Mode Helpers ──

var BlendModeUtil = (function() {
    'use strict';

    function toCompositeOp(mode: BlendModeKey): string {
        return BlendMode[mode] || BlendMode.Normal;
    }

    function allKeys(): BlendModeKey[] {
        return Object.keys(BlendMode) as BlendModeKey[];
    }

    function labelFor(mode: BlendModeKey): string {
        return BLEND_MODE_LABELS[mode] || mode;
    }

    return {
        toCompositeOp: toCompositeOp,
        allKeys: allKeys,
        labelFor: labelFor,
    };
})();
