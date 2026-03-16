/**
 * Status Bar — SerenityFlow Canvas v2
 *
 * Bottom bar: cursor position, zoom%, active layer, generation status,
 * VRAM usage, loaded model. Updated in real-time.
 */

var CanvasStatusBar = (function() {
    'use strict';

    var _barEl: HTMLElement | null = null;
    var _cursorEl: HTMLElement | null = null;
    var _zoomEl: HTMLElement | null = null;
    var _layerEl: HTMLElement | null = null;
    var _genStatusEl: HTMLElement | null = null;
    var _vramEl: HTMLElement | null = null;
    var _modelEl: HTMLElement | null = null;

    var _genState: 'idle' | 'queued' | 'generating' | 'complete' = 'idle';
    var _vramMb: number = 0;
    var _modelName: string = '';

    function create(container: HTMLElement): void {
        if (_barEl) return;

        _barEl = document.createElement('div');
        _barEl.id = 'cv-status-bar';
        _barEl.className = 'cv-status-bar';
        _barEl.innerHTML =
            '<span class="cv-sb-item" id="cv-sb-cursor" title="Cursor position">0, 0</span>' +
            '<span class="cv-sb-sep">|</span>' +
            '<span class="cv-sb-item" id="cv-sb-zoom" title="Zoom">100%</span>' +
            '<span class="cv-sb-sep">|</span>' +
            '<span class="cv-sb-item cv-sb-layer" id="cv-sb-layer" title="Active layer">—</span>' +
            '<span class="cv-sb-spacer"></span>' +
            '<span class="cv-sb-item cv-sb-gen" id="cv-sb-gen" title="Generation status">Idle</span>' +
            '<span class="cv-sb-sep">|</span>' +
            '<span class="cv-sb-item" id="cv-sb-vram" title="VRAM usage">—</span>' +
            '<span class="cv-sb-sep">|</span>' +
            '<span class="cv-sb-item cv-sb-model" id="cv-sb-model" title="Loaded model">No model</span>';

        container.appendChild(_barEl);

        _cursorEl = document.getElementById('cv-sb-cursor');
        _zoomEl = document.getElementById('cv-sb-zoom');
        _layerEl = document.getElementById('cv-sb-layer');
        _genStatusEl = document.getElementById('cv-sb-gen');
        _vramEl = document.getElementById('cv-sb-vram');
        _modelEl = document.getElementById('cv-sb-model');
    }

    function updateCursor(x: number, y: number): void {
        if (_cursorEl) _cursorEl.textContent = Math.round(x) + ', ' + Math.round(y);
    }

    function updateZoom(percent: number): void {
        if (_zoomEl) _zoomEl.textContent = percent + '%';
    }

    function updateActiveLayer(name: string, type: string): void {
        if (!_layerEl) return;
        var rawLabel = LAYER_TYPE_LABELS[type as LayerTypeValue] || type || 'Layer';
        _layerEl.textContent = rawLabel.toUpperCase() + ': ' + (name || 'Untitled');
    }

    function updateGenStatus(status: 'idle' | 'queued' | 'generating' | 'complete'): void {
        _genState = status || 'idle';
        if (!_genStatusEl) return;
        var labels: Record<string, string> = {
            idle: 'Idle',
            queued: 'Queued',
            generating: 'Generating...',
            complete: 'Complete',
        };
        _genStatusEl.textContent = labels[_genState] || String(_genState);
        _genStatusEl.className = 'cv-sb-item cv-sb-gen cv-sb-gen-' + _genState;
    }

    function updateVram(usedMb: number): void {
        _vramMb = usedMb;
        if (_vramEl) {
            if (usedMb <= 0) {
                _vramEl.textContent = '—';
            } else {
                var gb = (usedMb / 1024).toFixed(1);
                _vramEl.textContent = gb + ' GB';
            }
        }
    }

    function updateModel(name: string): void {
        _modelName = name;
        if (_modelEl) {
            _modelEl.textContent = name ? name.split('/').pop()!.replace(/\.\w+$/, '') : 'No model';
        }
    }

    function getGenStatus(): string { return _genState; }
    function getVramMb(): number { return _vramMb; }
    function getModelName(): string { return _modelName; }

    return {
        create: create,
        updateCursor: updateCursor,
        updateZoom: updateZoom,
        updateActiveLayer: updateActiveLayer,
        updateGenStatus: updateGenStatus,
        updateVram: updateVram,
        updateModel: updateModel,
        getGenStatus: getGenStatus,
        getVramMb: getVramMb,
        getModelName: getModelName,
    };
})();
