/**
 * Generate Tab — SerenityFlow Phase 2
 * Prompt-to-image generation wired to ComfyUI-compatible backend.
 */

var GenerateTab = (function() {
    'use strict';

    var state = {
        model: null,
        prompt: '',
        negPrompt: '',
        width: 1024,
        height: 1024,
        steps: 20,
        cfg: 7.0,
        guidance: 3.5,
        scheduler: 'euler',
        seed: -1,
        generating: false,
        currentImage: null,
        currentIsVideo: false,
        gallery: [],
        arch: 'sd15',
        frames: 97,
        fps: 24,
        lastSeed: null,
        batchCount: 1,
        loras: [],  // [{name, strength}]
        aspectLocked: false,
        lockedRatio: 1
    };

    var initialized = false;

    // DOM refs (set in init)
    var els = {};

    // ── Aspect ratio definitions ──
    var imageAspects = [
        { label: '1:1',  w: 1024, h: 1024, vw: 16, vh: 16 },
        { label: '4:3',  w: 1152, h: 896,  vw: 18, vh: 14 },
        { label: '16:9', w: 1344, h: 768,  vw: 20, vh: 11 },
        { label: '3:4',  w: 896,  h: 1152, vw: 14, vh: 18 },
        { label: '9:16', w: 768,  h: 1344, vw: 11, vh: 20 }
    ];

    var videoAspects = [
        { label: '1:1',  w: 512, h: 512, vw: 16, vh: 16 },
        { label: '4:3',  w: 768, h: 576, vw: 18, vh: 14 },
        { label: '16:9', w: 768, h: 432, vw: 20, vh: 11 },
        { label: '3:4',  w: 576, h: 768, vw: 14, vh: 18 },
        { label: '9:16', w: 432, h: 768, vw: 11, vh: 20 }
    ];

    function getActiveAspects() {
        return ModelUtils.isVideoModel(state.model) ? videoAspects : imageAspects;
    }

    // ── Build DOM ──
    function buildUI() {
        var panel = document.getElementById('panel-generate');
        if (!panel) return;
        panel.innerHTML = '';

        var layout = document.createElement('div');
        layout.className = 'gen-layout';

        // Left panel
        var left = document.createElement('div');
        left.className = 'gen-left';
        left.innerHTML = buildLeftHTML();
        layout.appendChild(left);

        // Center panel
        var center = document.createElement('div');
        center.className = 'gen-center';
        center.innerHTML = buildCenterHTML();
        layout.appendChild(center);

        // Right panel
        var right = document.createElement('div');
        right.className = 'gen-right';
        right.innerHTML = buildRightHTML();
        layout.appendChild(right);

        panel.appendChild(layout);
        cacheElements();
    }

    function buildLeftHTML() {
        return '' +
        // Model
        '<div class="gen-section">' +
            '<label class="gen-label">Model</label>' +
            '<select id="gen-model" class="gen-select"><option disabled selected>Loading models...</option></select>' +
            '<div id="gen-model-warn" class="gen-model-warning"></div>' +
        '</div>' +

        // Prompt
        '<div class="gen-section">' +
            '<label class="gen-label">Prompt</label>' +
            '<textarea id="gen-prompt" class="gen-textarea" rows="4" placeholder="Describe your image..."></textarea>' +
            '<div id="gen-token-count" class="gen-token-count">~0 tokens</div>' +
        '</div>' +

        // Negative prompt
        '<div class="gen-section" id="gen-neg-section">' +
            '<label class="gen-label">Negative Prompt</label>' +
            '<textarea id="gen-neg-prompt" class="gen-textarea" rows="2" placeholder="What to avoid..."></textarea>' +
        '</div>' +

        // LoRA
        '<div class="gen-section" id="gen-lora-section">' +
            '<label class="gen-label">LoRA</label>' +
            '<div id="gen-lora-list" class="gen-lora-list"></div>' +
            '<button id="gen-lora-add" class="gen-lora-add-btn">+ Add LoRA</button>' +
            '<select id="gen-lora-picker" class="gen-select" style="display:none">' +
                '<option disabled selected>Select LoRA...</option>' +
            '</select>' +
        '</div>' +

        // Aspect ratio
        '<div class="gen-section">' +
            '<label class="gen-label">Aspect Ratio</label>' +
            '<div id="gen-aspects" class="gen-aspect-grid"></div>' +
            '<div class="gen-lock-row">' +
                '<button id="gen-aspect-lock" class="gen-aspect-lock" title="Lock aspect ratio">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>' +
                '</button>' +
            '</div>' +
            '<div class="gen-custom-res">' +
                '<div class="gen-res-row">' +
                    '<label class="gen-res-label">W</label>' +
                    '<input type="number" id="gen-custom-width" class="gen-number-input" min="256" max="4096" step="64" value="1024">' +
                '</div>' +
                '<div class="gen-res-row">' +
                    '<label class="gen-res-label">H</label>' +
                    '<input type="number" id="gen-custom-height" class="gen-number-input" min="256" max="4096" step="64" value="1024">' +
                '</div>' +
            '</div>' +
        '</div>' +

        // Video controls (hidden by default, shown for video models)
        '<div class="gen-section" id="gen-video-section" style="display:none">' +
            '<label class="gen-label">Video</label>' +
            '<div class="gen-setting-row">' +
                '<span class="gen-label" style="min-width:52px;margin-bottom:0">Frames</span>' +
                '<input type="number" id="gen-frames" class="gen-number-input" min="9" max="257" step="8" value="97">' +
                '<input type="range" id="gen-frames-range" class="gen-range" min="9" max="257" step="8" value="97">' +
            '</div>' +
            '<div class="gen-setting-row">' +
                '<span class="gen-label" style="min-width:52px;margin-bottom:0">FPS</span>' +
                '<input type="number" id="gen-fps" class="gen-number-input" min="8" max="60" step="1" value="24">' +
                '<input type="range" id="gen-fps-range" class="gen-range" min="8" max="60" step="1" value="24">' +
            '</div>' +
            '<div id="gen-duration-hint" class="gen-duration-hint"></div>' +
        '</div>' +

        // Settings accordion
        '<div class="gen-section">' +
            '<div id="gen-settings-header" class="gen-accordion-header">' +
                '<span>Settings</span>' +
                '<span class="gen-accordion-arrow">&#9660;</span>' +
            '</div>' +
            '<div id="gen-settings-body" class="gen-accordion-body" style="margin-top:8px">' +
                // Steps
                '<div class="gen-setting-row">' +
                    '<span class="gen-label">Steps</span>' +
                    '<input id="gen-steps" type="number" class="gen-number-input" min="1" max="150" value="20">' +
                    '<input id="gen-steps-range" type="range" class="gen-range" min="1" max="150" value="20">' +
                '</div>' +
                // CFG
                '<div id="gen-cfg-row" class="gen-setting-row">' +
                    '<span class="gen-label">CFG</span>' +
                    '<input id="gen-cfg" type="number" class="gen-number-input" min="1" max="20" step="0.5" value="7.0">' +
                    '<input id="gen-cfg-range" type="range" class="gen-range" min="1" max="20" step="0.5" value="7.0">' +
                '</div>' +
                // Guidance (FLUX only)
                '<div id="gen-guidance-row" class="gen-setting-row" style="display:none">' +
                    '<span class="gen-label">Guidance</span>' +
                    '<input id="gen-guidance" type="number" class="gen-number-input" min="1" max="10" step="0.5" value="3.5">' +
                    '<input id="gen-guidance-range" type="range" class="gen-range" min="1" max="10" step="0.5" value="3.5">' +
                '</div>' +
                // Scheduler
                '<div class="gen-setting-row">' +
                    '<span class="gen-label">Sampler</span>' +
                    '<select id="gen-scheduler" class="gen-select" style="flex:1">' +
                        '<option value="euler" title="Fast, general-purpose sampler">euler</option>' +
                        '<option value="euler_ancestral" title="Adds noise each step for more creative results">euler_ancestral</option>' +
                        '<option value="dpm_2" title="Second-order DPM solver, good quality">dpm_2</option>' +
                        '<option value="dpm_2_ancestral" title="DPM-2 with ancestral sampling for variety">dpm_2_ancestral</option>' +
                        '<option value="dpmpp_2m" title="Fast multistep solver, good for fewer steps">dpm++_2m</option>' +
                        '<option value="dpmpp_sde" title="Stochastic solver, great detail at higher steps">dpm++_sde</option>' +
                        '<option value="ddim" title="Deterministic sampler, supports image-to-image well">ddim</option>' +
                        '<option value="lcm" title="Latent consistency model, very fast (4-8 steps)">lcm</option>' +
                    '</select>' +
                '</div>' +
            '</div>' +
        '</div>' +

        // Seed
        '<div class="gen-section">' +
            '<label class="gen-label">Seed</label>' +
            '<div class="gen-seed-row">' +
                '<input id="gen-seed" type="number" class="gen-number-input" value="-1">' +
                '<button id="gen-seed-shuffle" class="gen-seed-shuffle" title="Random seed">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 18h1.4c1.3 0 2.5-.6 3.3-1.7l6.1-8.6c.7-1.1 2-1.7 3.3-1.7H22"/><path d="m18 2 4 4-4 4"/><path d="M2 6h1.9c1.5 0 2.9.9 3.6 2.2"/><path d="M22 18h-5.9c-1.3 0-2.6-.7-3.3-1.8l-.5-.8"/><path d="m18 14 4 4-4 4"/></svg>' +
                '</button>' +
                '<button id="gen-seed-prev" class="gen-seed-shuffle" title="Use previous seed" style="font-size:12px">&#8634;</button>' +
            '</div>' +
        '</div>' +

        // Batch
        '<div class="gen-section">' +
            '<label class="gen-label">Batch</label>' +
            '<div class="gen-batch-row">' +
                '<input id="gen-batch" type="number" class="gen-number-input" min="1" max="8" value="1">' +
                '<span class="gen-batch-hint">images per run</span>' +
            '</div>' +
        '</div>' +

        // Generate button
        '<div class="gen-section">' +
            '<button id="gen-btn" class="gen-btn">Generate</button>' +
        '</div>';
    }

    function buildCenterHTML() {
        return '' +
            '<div id="gen-empty" class="gen-empty">' +
                '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>' +
                '<span class="gen-empty-text">Your image will appear here</span>' +
            '</div>' +
            '<img id="gen-preview-img" class="gen-preview-img" style="display:none" alt="Generated image">' +
            '<video id="gen-preview-video" class="gen-preview-video" style="display:none" autoplay loop muted playsinline controls></video>' +
            '<div id="gen-action-bar" class="gen-action-bar" style="display:none">' +
                '<button class="gen-action-btn" id="gen-download" title="Download"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg></button>' +
                '<button class="gen-action-btn" id="gen-to-canvas" title="Coming in Canvas tab" disabled><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg></button>' +
                '<button class="gen-action-btn" id="gen-clear-preview" title="Clear"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg></button>' +
            '</div>' +
            '<div id="gen-progress-label" class="gen-progress-label"></div>' +
            '<div id="gen-progress" class="gen-progress"><div id="gen-progress-bar" class="gen-progress-bar"></div></div>' +
            '<div id="gen-error-banner" class="gen-error-banner"></div>' +
            '<div id="gen-ws-indicator" class="gen-ws-indicator"><span class="gen-ws-dot"></span><span>Reconnecting...</span></div>';
    }

    function buildRightHTML() {
        return '' +
            '<div class="gen-gallery-header">' +
                '<span class="gen-gallery-title">Gallery</span>' +
                '<button id="gen-gallery-clear" class="gen-gallery-clear">Clear</button>' +
            '</div>' +
            '<div id="gen-gallery-grid" class="gen-gallery-grid"></div>';
    }

    function cacheElements() {
        els.model = document.getElementById('gen-model');
        els.modelWarn = document.getElementById('gen-model-warn');
        els.prompt = document.getElementById('gen-prompt');
        els.negPrompt = document.getElementById('gen-neg-prompt');
        els.negSection = document.getElementById('gen-neg-section');
        els.aspectGrid = document.getElementById('gen-aspects');
        els.customWidth = document.getElementById('gen-custom-width');
        els.customHeight = document.getElementById('gen-custom-height');
        els.videoSection = document.getElementById('gen-video-section');
        els.framesInput = document.getElementById('gen-frames');
        els.framesRange = document.getElementById('gen-frames-range');
        els.fpsInput = document.getElementById('gen-fps');
        els.fpsRange = document.getElementById('gen-fps-range');
        els.durationHint = document.getElementById('gen-duration-hint');
        els.settingsHeader = document.getElementById('gen-settings-header');
        els.settingsBody = document.getElementById('gen-settings-body');
        els.steps = document.getElementById('gen-steps');
        els.stepsRange = document.getElementById('gen-steps-range');
        els.cfgRow = document.getElementById('gen-cfg-row');
        els.cfg = document.getElementById('gen-cfg');
        els.cfgRange = document.getElementById('gen-cfg-range');
        els.guidanceRow = document.getElementById('gen-guidance-row');
        els.guidance = document.getElementById('gen-guidance');
        els.guidanceRange = document.getElementById('gen-guidance-range');
        els.scheduler = document.getElementById('gen-scheduler');
        els.seed = document.getElementById('gen-seed');
        els.seedShuffle = document.getElementById('gen-seed-shuffle');
        els.btn = document.getElementById('gen-btn');
        els.empty = document.getElementById('gen-empty');
        els.previewImg = document.getElementById('gen-preview-img');
        els.previewVideo = document.getElementById('gen-preview-video');
        els.actionBar = document.getElementById('gen-action-bar');
        els.download = document.getElementById('gen-download');
        els.clearPreview = document.getElementById('gen-clear-preview');
        els.progress = document.getElementById('gen-progress');
        els.progressBar = document.getElementById('gen-progress-bar');
        els.progressLabel = document.getElementById('gen-progress-label');
        els.errorBanner = document.getElementById('gen-error-banner');
        els.wsIndicator = document.getElementById('gen-ws-indicator');
        els.galleryGrid = document.getElementById('gen-gallery-grid');
        els.galleryClear = document.getElementById('gen-gallery-clear');
    }

    // ── Aspect Ratio Buttons ──
    function buildAspectButtons() {
        var aspects = getActiveAspects();
        els.aspectGrid.innerHTML = '';
        aspects.forEach(function(a, i) {
            var btn = document.createElement('button');
            btn.className = 'gen-aspect-btn' + (a.w === state.width && a.h === state.height ? ' active' : '');
            var maxDim = 20;
            var sw = Math.round(a.vw / maxDim * 16);
            var sh = Math.round(a.vh / maxDim * 16);
            btn.innerHTML =
                '<svg width="' + (sw + 4) + '" height="' + (sh + 4) + '" viewBox="0 0 ' + (sw + 4) + ' ' + (sh + 4) + '" fill="none" stroke-width="1.5">' +
                    '<rect x="1" y="1" width="' + sw + '" height="' + sh + '" rx="2"/>' +
                '</svg>' +
                '<span class="gen-aspect-label">' + a.label + '</span>';
            btn.addEventListener('click', function() {
                state.width = a.w;
                state.height = a.h;
                els.customWidth.value = a.w;
                els.customHeight.value = a.h;
                els.aspectGrid.querySelectorAll('.gen-aspect-btn').forEach(function(b) {
                    b.classList.remove('active');
                });
                btn.classList.add('active');
            });
            els.aspectGrid.appendChild(btn);
        });
    }

    // ── Event Binding ──
    function bindEvents() {
        // Auto-grow prompt textarea
        els.prompt.addEventListener('input', function() {
            state.prompt = this.value;
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
            updateTokenCount();
        });
        els.negPrompt.addEventListener('input', function() {
            state.negPrompt = this.value;
        });

        // Settings accordion
        els.settingsHeader.addEventListener('click', function() {
            this.classList.toggle('closed');
            els.settingsBody.classList.toggle('closed');
        });

        // Steps sync
        els.steps.addEventListener('input', function() {
            state.steps = parseInt(this.value) || 20;
            els.stepsRange.value = this.value;
        });
        els.stepsRange.addEventListener('input', function() {
            state.steps = parseInt(this.value);
            els.steps.value = this.value;
        });

        // CFG sync
        els.cfg.addEventListener('input', function() {
            state.cfg = parseFloat(this.value) || 7.0;
            els.cfgRange.value = this.value;
        });
        els.cfgRange.addEventListener('input', function() {
            state.cfg = parseFloat(this.value);
            els.cfg.value = this.value;
        });

        // Scheduler
        els.scheduler.addEventListener('change', function() {
            state.scheduler = this.value;
        });

        // Model
        els.model.addEventListener('change', function() {
            state.model = this.value;
            updateUIForArch(ModelUtils.detectArchFromFilename(this.value));
        });

        // Guidance sync (FLUX)
        els.guidance.addEventListener('input', function() {
            state.guidance = parseFloat(this.value) || 3.5;
            els.guidanceRange.value = this.value;
        });
        els.guidanceRange.addEventListener('input', function() {
            state.guidance = parseFloat(this.value);
            els.guidance.value = this.value;
        });

        // Aspect ratio lock
        var lockBtn = document.getElementById('gen-aspect-lock');
        if (lockBtn) {
            lockBtn.addEventListener('click', function() {
                state.aspectLocked = !state.aspectLocked;
                this.classList.toggle('active', state.aspectLocked);
                if (state.aspectLocked) {
                    state.lockedRatio = state.width / state.height;
                }
            });
        }

        // Custom resolution inputs
        els.customWidth.addEventListener('blur', function() {
            var isVideo = ModelUtils.isVideoModel(state.model);
            var v = isVideo
                ? ModelUtils.clampVideoDimension(parseInt(this.value) || 512)
                : ModelUtils.clampDimension(parseInt(this.value) || 1024);
            this.value = v;
            state.width = v;
            if (state.aspectLocked && state.lockedRatio) {
                var newH = isVideo
                    ? ModelUtils.clampVideoDimension(Math.round(v / state.lockedRatio))
                    : ModelUtils.clampDimension(Math.round(v / state.lockedRatio));
                state.height = newH;
                els.customHeight.value = newH;
            }
            deselectAspectButtons();
        });
        els.customHeight.addEventListener('blur', function() {
            var isVideo = ModelUtils.isVideoModel(state.model);
            var v = isVideo
                ? ModelUtils.clampVideoDimension(parseInt(this.value) || 512)
                : ModelUtils.clampDimension(parseInt(this.value) || 1024);
            this.value = v;
            state.height = v;
            if (state.aspectLocked && state.lockedRatio) {
                var newW = isVideo
                    ? ModelUtils.clampVideoDimension(Math.round(v * state.lockedRatio))
                    : ModelUtils.clampDimension(Math.round(v * state.lockedRatio));
                state.width = newW;
                els.customWidth.value = newW;
            }
            deselectAspectButtons();
        });

        // Seed
        els.seed.addEventListener('input', function() {
            state.seed = parseInt(this.value);
        });
        els.seedShuffle.addEventListener('click', function() {
            var s = Math.floor(Math.random() * 4294967296);
            state.seed = s;
            els.seed.value = s;
        });

        // Seed previous
        var seedPrev = document.getElementById('gen-seed-prev');
        if (seedPrev) {
            seedPrev.addEventListener('click', function() {
                if (state.lastSeed !== null) {
                    state.seed = state.lastSeed;
                    els.seed.value = state.lastSeed;
                }
            });
        }

        // Batch count
        var batchInput = document.getElementById('gen-batch');
        if (batchInput) {
            batchInput.addEventListener('input', function() {
                state.batchCount = Math.max(1, Math.min(8, parseInt(this.value) || 1));
            });
        }

        // LoRA add/picker
        var loraAdd = document.getElementById('gen-lora-add');
        var loraPicker = document.getElementById('gen-lora-picker');
        if (loraAdd && loraPicker) {
            loraAdd.addEventListener('click', function() {
                loraPicker.style.display = loraPicker.style.display === 'none' ? 'block' : 'none';
            });
            loraPicker.addEventListener('change', function() {
                if (this.value) {
                    addLora(this.value);
                    this.selectedIndex = 0;
                    this.style.display = 'none';
                }
            });
        }

        // Frames sync
        els.framesInput.addEventListener('input', function() {
            state.frames = parseInt(this.value) || 97;
            els.framesRange.value = this.value;
            updateDurationHint();
        });
        els.framesRange.addEventListener('input', function() {
            state.frames = parseInt(this.value);
            els.framesInput.value = this.value;
            updateDurationHint();
        });

        // FPS sync
        els.fpsInput.addEventListener('input', function() {
            state.fps = parseInt(this.value) || 24;
            els.fpsRange.value = this.value;
            updateDurationHint();
        });
        els.fpsRange.addEventListener('input', function() {
            state.fps = parseInt(this.value);
            els.fpsInput.value = this.value;
            updateDurationHint();
        });

        // Generate
        els.btn.addEventListener('click', function() {
            generate();
        });

        // Action bar
        els.download.addEventListener('click', function() {
            if (!state.currentImage) return;
            var a = document.createElement('a');
            a.href = state.currentImage;
            var ext = state.currentIsVideo ? '.mp4' : '.png';
            a.download = 'serenityflow_' + Date.now() + ext;
            a.click();
        });
        els.clearPreview.addEventListener('click', function() {
            clearPreview();
        });

        // Gallery clear
        els.galleryClear.addEventListener('click', function() {
            state.gallery = [];
            els.galleryGrid.innerHTML = '';
            localStorage.removeItem('sf-gallery');
        });
    }

    // ── Model Loading ──
    function loadModels() {
        ModelUtils.fetchAllModels()
            .then(function(models) {
                if (!models.length) throw new Error('empty');

                els.model.innerHTML = '';
                models.forEach(function(m) {
                    var opt = document.createElement('option');
                    opt.value = m.name;
                    opt.textContent = m.name;
                    els.model.appendChild(opt);
                });
                state.model = models[0].name;
                updateUIForArch(ModelUtils.detectArchFromFilename(models[0].name));
                els.modelWarn.classList.remove('visible');
            })
            .catch(function() {
                els.model.innerHTML = '<option disabled selected>No models found</option>';
                els.modelWarn.textContent = 'Could not load models \u2014 is the server running?';
                els.modelWarn.classList.add('visible');
            });
    }

    // ── Arch-aware UI ──
    function updateUIForArch(arch) {
        state.arch = arch;
        var isFlux = arch === 'flux';
        var isVideo = arch === 'ltxv' || arch === 'wan';

        // Neg prompt: hide for flux and video
        els.negSection.style.display = (isFlux || isVideo) ? 'none' : 'block';
        // CFG: hide for flux and video
        els.cfgRow.style.display = (isFlux || isVideo) ? 'none' : 'flex';
        // Guidance: flux only
        els.guidanceRow.style.display = isFlux ? 'flex' : 'none';
        // Video controls
        els.videoSection.style.display = isVideo ? 'block' : 'none';

        // Batch: hide for video models
        var batchInput = document.getElementById('gen-batch');
        if (batchInput) batchInput.parentElement.parentElement.style.display = isVideo ? 'none' : 'block';

        // Button label
        els.btn.textContent = isVideo ? 'Generate Video' : 'Generate';

        // Rebuild aspect ratio buttons for the right set
        buildAspectButtons();

        // Update custom res input constraints
        if (isVideo) {
            els.customWidth.min = 64;
            els.customWidth.max = 1280;
            els.customWidth.step = 32;
            els.customHeight.min = 64;
            els.customHeight.max = 1280;
            els.customHeight.step = 32;
        } else {
            els.customWidth.min = 256;
            els.customWidth.max = 4096;
            els.customWidth.step = 64;
            els.customHeight.min = 256;
            els.customHeight.max = 4096;
            els.customHeight.step = 64;
        }

        // Select first aspect ratio for the new mode
        var aspects = getActiveAspects();
        if (aspects.length > 0) {
            state.width = aspects[0].w;
            state.height = aspects[0].h;
            els.customWidth.value = aspects[0].w;
            els.customHeight.value = aspects[0].h;
        }

        if (isVideo) updateDurationHint();
    }

    function updateDurationHint() {
        if (!els.durationHint) return;
        var secs = (state.frames / state.fps).toFixed(1);
        els.durationHint.textContent = '\u2248 ' + secs + 's at ' + state.fps + 'fps';
    }

    function deselectAspectButtons() {
        els.aspectGrid.querySelectorAll('.gen-aspect-btn').forEach(function(b) {
            b.classList.remove('active');
        });
    }

    // ── Token Counter ──
    function updateTokenCount() {
        var count = state.prompt.trim() ? state.prompt.trim().split(/\s+/).length : 0;
        var el = document.getElementById('gen-token-count');
        if (el) el.textContent = '~' + count + ' tokens';
    }

    // ── LoRA Management ──
    function loadLoras() {
        fetch('/models/loras')
            .then(function(r) { return r.ok ? r.json() : []; })
            .then(function(list) {
                if (!list || !list.length) return;
                var picker = document.getElementById('gen-lora-picker');
                if (!picker) return;
                picker.innerHTML = '<option disabled selected>Select LoRA...</option>';
                list.forEach(function(name) {
                    var opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = name;
                    picker.appendChild(opt);
                });
            })
            .catch(function() {});
    }

    function addLora(name) {
        if (state.loras.some(function(l) { return l.name === name; })) return;
        state.loras.push({ name: name, strength: 1.0 });
        renderLoraList();
    }

    function removeLora(idx) {
        state.loras.splice(idx, 1);
        renderLoraList();
    }

    function renderLoraList() {
        var list = document.getElementById('gen-lora-list');
        if (!list) return;
        list.innerHTML = '';
        state.loras.forEach(function(lora, idx) {
            var row = document.createElement('div');
            row.className = 'gen-lora-row';
            row.innerHTML =
                '<span class="gen-lora-name">' + lora.name + '</span>' +
                '<input type="range" class="gen-range gen-lora-strength" min="-2" max="2" step="0.05" value="' + lora.strength + '">' +
                '<span class="gen-lora-val">' + lora.strength.toFixed(2) + '</span>' +
                '<button class="gen-lora-remove" data-idx="' + idx + '">&times;</button>';
            list.appendChild(row);
        });
        // Bind events
        list.querySelectorAll('.gen-lora-strength').forEach(function(slider, idx) {
            slider.addEventListener('input', function() {
                state.loras[idx].strength = parseFloat(this.value);
                this.nextElementSibling.textContent = state.loras[idx].strength.toFixed(2);
            });
        });
        list.querySelectorAll('.gen-lora-remove').forEach(function(btn) {
            btn.addEventListener('click', function() {
                removeLora(parseInt(this.dataset.idx));
            });
        });
    }

    // ── Workflow Builder ──
    function buildWorkflow() {
        return WorkflowBuilder.build({
            model: state.model,
            prompt: state.prompt,
            negPrompt: state.negPrompt,
            width: state.width,
            height: state.height,
            steps: state.steps,
            cfg: state.cfg,
            guidance: state.guidance,
            scheduler: state.scheduler,
            seed: state.seed,
            frames: state.frames,
            fps: state.fps,
            loras: state.loras
        });
    }

    // ── Generate ──
    function generate() {
        if (state.generating) return;
        if (!state.model) {
            showError('No model selected');
            return;
        }
        if (!state.prompt.trim()) {
            showError('Enter a prompt');
            return;
        }

        setGenerating(true);

        var batchN = state.batchCount || 1;
        var resolvedSeed = state.seed === -1 ? Math.floor(Math.random() * 4294967296) : state.seed;
        state.lastSeed = resolvedSeed;

        // Build and queue all batch workflows upfront
        var originalSeed = state.seed;
        var workflows = [];
        for (var i = 0; i < batchN; i++) {
            state.seed = (i === 0) ? resolvedSeed : Math.floor(Math.random() * 4294967296);
            workflows.push(buildWorkflow());
        }
        state.seed = originalSeed;

        // Queue all batch items (server handles sequencing)
        var queueFailed = false;
        workflows.forEach(function(workflow, i) {
            if (queueFailed) return;
            if (batchN > 1) {
                els.btn.textContent = 'Generating ' + (i + 1) + ' / ' + batchN + '...';
            }
            SerenityAPI.postPrompt(workflow, {
                prompt: state.prompt,
                model: state.model,
                batchLabel: batchN > 1 ? ('(' + (i + 1) + '/' + batchN + ')') : ''
            })
            .catch(function(err) {
                queueFailed = true;
                showError('Failed to queue: ' + err.message);
                setGenerating(false);
            });
        });
    }

    // ── State Helpers ──
    function setGenerating(v) {
        state.generating = v;
        var isVideo = ModelUtils.isVideoModel(state.model);
        els.btn.disabled = v;
        if (v) {
            els.btn.textContent = 'Generating...';
        } else {
            els.btn.textContent = isVideo ? 'Generate Video' : 'Generate';
        }
        els.btn.classList.toggle('generating', v);
        if (v) {
            els.progress.classList.add('active');
            els.progressBar.style.width = '100%';
        } else {
            els.progress.classList.remove('active');
            els.progressBar.style.width = '0%';
            els.progressLabel.classList.remove('visible');
        }
    }

    function displayImage(src) {
        state.currentImage = src;
        state.currentIsVideo = false;
        els.previewImg.src = src;
        els.previewImg.style.display = 'block';
        els.previewVideo.style.display = 'none';
        els.previewVideo.pause();
        els.actionBar.style.display = 'flex';
        els.empty.style.display = 'none';
    }

    function displayVideo(src) {
        state.currentImage = src;
        state.currentIsVideo = true;
        els.previewVideo.src = src;
        els.previewVideo.style.display = 'block';
        els.previewImg.style.display = 'none';
        els.actionBar.style.display = 'flex';
        els.empty.style.display = 'none';
    }

    function clearPreview() {
        state.currentImage = null;
        state.currentIsVideo = false;
        els.previewImg.style.display = 'none';
        els.previewImg.removeAttribute('src');
        els.previewVideo.style.display = 'none';
        els.previewVideo.pause();
        els.previewVideo.removeAttribute('src');
        els.actionBar.style.display = 'none';
        els.empty.style.display = 'flex';
        // Deselect gallery thumbs
        els.galleryGrid.querySelectorAll('.gen-thumb-wrap').forEach(function(t) {
            t.classList.remove('active');
        });
    }

    function addToGallery(src, isVideo) {
        state.gallery.unshift({ src: src, isVideo: !!isVideo });
        if (state.gallery.length > 200) state.gallery.pop();

        var wrap = createThumb(src, !!isVideo, true);
        if (els.galleryGrid.firstChild) {
            els.galleryGrid.insertBefore(wrap, els.galleryGrid.firstChild);
        } else {
            els.galleryGrid.appendChild(wrap);
        }

        // Trim DOM to 200
        while (els.galleryGrid.children.length > 200) {
            els.galleryGrid.removeChild(els.galleryGrid.lastChild);
        }

        saveGallery();
    }

    function createThumb(src, isVideo, isActive) {
        var wrap = document.createElement('div');
        wrap.className = 'gen-thumb-wrap' + (isActive ? ' active' : '') + (isVideo ? ' gen-thumb-video' : '');
        if (isVideo) {
            wrap.innerHTML =
                '<video src="' + src + '#t=0.1" muted preload="metadata" style="width:100%;height:100%;object-fit:cover;display:block;"></video>' +
                '<div class="gen-thumb-play">\u25b6</div>';
        } else {
            wrap.innerHTML =
                '<img src="' + src + '" alt="thumbnail" loading="lazy">' +
                '<div class="gen-thumb-overlay">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>' +
                '</div>';
        }
        wrap.addEventListener('click', function() {
            if (isVideo) {
                displayVideo(src);
            } else {
                displayImage(src);
            }
            els.galleryGrid.querySelectorAll('.gen-thumb-wrap').forEach(function(t) {
                t.classList.remove('active');
            });
            wrap.classList.add('active');
        });
        return wrap;
    }

    function saveGallery() {
        try {
            localStorage.setItem('sf-gallery', JSON.stringify(state.gallery));
        } catch(e) { /* quota exceeded */ }
    }

    function restoreGallery() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-gallery'));
            if (saved && Array.isArray(saved)) {
                // Handle both old (string[]) and new ({src, isVideo}[]) formats
                state.gallery = saved.slice(0, 200).map(function(item) {
                    if (typeof item === 'string') return { src: item, isVideo: false };
                    return item;
                });
                state.gallery.forEach(function(item) {
                    els.galleryGrid.appendChild(createThumb(item.src, item.isVideo, false));
                });
            }
        } catch(e) { /* ignore */ }
    }

    // ── Error Display ──
    function showError(msg) {
        els.errorBanner.textContent = msg;
        els.errorBanner.classList.add('visible');
        setTimeout(function() {
            els.errorBanner.classList.remove('visible');
        }, 5000);
    }

    // ── WebSocket (via shared SerenityWS) ──
    function connectWS() {
        SerenityWS.on('connected', function() {
            els.wsIndicator.classList.remove('visible');
        });

        SerenityWS.on('disconnected', function() {
            els.wsIndicator.classList.add('visible');
        });

        SerenityWS.on('execution_start', function() {
            setGenerating(true);
        });

        SerenityWS.on('progress', function(data) {
            if (!data) return;
            var pct = (data.value / data.max * 100).toFixed(0);
            els.progressBar.style.width = pct + '%';
            els.progressLabel.textContent = 'Step ' + data.value + ' / ' + data.max;
            els.progressLabel.classList.add('visible');
        });

        SerenityWS.on('executed', function(data) {
            if (!data || !data.output) return;
            // Image outputs
            var out = data.output.ui || data.output;
            var items = out.images;
            var isVideo = false;
            // Video outputs (SaveAnimatedWEBP, SaveVideo)
            if (!items && out.videos) {
                items = out.videos;
                isVideo = true;
            }
            if (!items || !items.length) return;

            var file = items[0];
            var src = '/view?filename=' + encodeURIComponent(file.filename) +
                '&subfolder=' + encodeURIComponent(file.subfolder || '') +
                '&type=' + encodeURIComponent(file.type || 'output');
            // Also detect video from filename extension
            if (!isVideo) isVideo = /\.(webp|mp4|gif)$/i.test(file.filename);
            if (isVideo) {
                displayVideo(src);
            } else {
                displayImage(src);
            }
            addToGallery(src, isVideo);
            setGenerating(false);
        });

        SerenityWS.on('execution_error', function(data) {
            var errMsg = (data && data.exception_message) || 'Generation failed';
            showError(errMsg);
            setGenerating(false);
        });
    }

    // ── Public API ──
    function init() {
        if (initialized) return;
        initialized = true;

        buildUI();
        buildAspectButtons();
        bindEvents();
        loadModels();
        loadLoras();
        restoreGallery();
        connectWS();
    }

    return {
        state: state,
        init: init,
        generate: generate,
        displayResult: function(src, isVideo) {
            if (!initialized) init();
            if (isVideo) displayVideo(src);
            else displayImage(src);
        }
    };
})();
