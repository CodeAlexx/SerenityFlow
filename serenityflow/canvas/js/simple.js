/**
 * Simple Mode — SerenityFlow
 * Streamlined single-screen UI for average users.
 * Uses WorkflowBuilder for generation (same backend as Generate tab).
 */

var SimpleMode = (function() {
    'use strict';

    var initialized = false;

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
        recent: [],
        arch: 'sd15',
        frames: 97,
        fps: 24,
        activeStyle: 'none',
        quality: 'balanced',
        duration: 'medium'
    };

    var els = {};

    // ── Presets ──

    var qualityPresets = {
        draft:    { steps: 10 },
        balanced: { steps: 20 },
        quality:  { steps: 40 }
    };

    var durationPresets = {
        short:  { frames: 49,  fps: 24 },
        medium: { frames: 97,  fps: 24 },
        long:   { frames: 193, fps: 24 }
    };

    var stylePresets = [
        { id: 'none',   label: 'None',           suffix: '' },
        { id: 'photo',  label: 'Photorealistic',  suffix: ', photorealistic, high detail, 8k' },
        { id: 'anime',  label: 'Anime',           suffix: ', anime style, cel-shaded' },
        { id: 'oil',    label: 'Oil Painting',     suffix: ', oil painting, textured brushstrokes' },
        { id: '3d',     label: '3D Render',        suffix: ', 3D render, octane render' },
        { id: 'cinema', label: 'Cinematic',        suffix: ', cinematic lighting, film grain, dramatic' }
    ];

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

    function isVideoModel() {
        return ModelUtils.isVideoModel(state.model);
    }

    function getActiveAspects() {
        return isVideoModel() ? videoAspects : imageAspects;
    }

    // ── Build DOM ──

    function buildUI() {
        var container = document.getElementById('simple-mode-container');
        if (!container) return;
        container.innerHTML = '';

        var layout = document.createElement('div');
        layout.className = 'simple-layout';

        var left = document.createElement('div');
        left.className = 'simple-left';
        left.innerHTML = buildLeftHTML();
        layout.appendChild(left);

        var center = document.createElement('div');
        center.className = 'simple-center';
        center.innerHTML = buildCenterHTML();
        layout.appendChild(center);

        container.appendChild(layout);
        cacheElements();
    }

    function buildLeftHTML() {
        // Style pills HTML
        var stylePills = '';
        stylePresets.forEach(function(s) {
            var active = s.id === state.activeStyle ? ' active' : '';
            stylePills += '<button class="simple-style-pill' + active + '" data-style="' + s.id + '">' + s.label + '</button>';
        });

        return '' +
        // Model
        '<div class="simple-section">' +
            '<label class="simple-label">Model</label>' +
            '<select id="simple-model" class="simple-select"><option disabled selected>Loading models...</option></select>' +
        '</div>' +

        // Prompt
        '<div class="simple-section">' +
            '<label class="simple-label">Prompt</label>' +
            '<textarea id="simple-prompt" class="simple-prompt" rows="5" placeholder="Describe what you want to create..."></textarea>' +
            '<div id="simple-adv-toggle" class="simple-disclosure">' +
                '<span class="simple-disclosure-arrow">&#9654;</span> Advanced prompt' +
            '</div>' +
            '<div id="simple-adv-body" class="simple-disclosure-body">' +
                '<textarea id="simple-neg-prompt" class="simple-neg-prompt" rows="2" placeholder="What to avoid..."></textarea>' +
                '<div class="simple-styles" id="simple-styles">' + stylePills + '</div>' +
            '</div>' +
        '</div>' +

        // Quick Settings
        '<div class="simple-section simple-quick">' +
            '<span class="simple-quick-label">Aspect Ratio</span>' +
            '<div id="simple-aspects" class="simple-aspect-grid"></div>' +
            '<span class="simple-quick-label">Quality</span>' +
            '<div id="simple-quality" class="simple-quality-row">' +
                '<button class="simple-quality-btn" data-quality="draft">Draft</button>' +
                '<button class="simple-quality-btn active" data-quality="balanced">Balanced</button>' +
                '<button class="simple-quality-btn" data-quality="quality">Quality</button>' +
            '</div>' +
            '<div id="simple-duration-section" class="simple-duration-section">' +
                '<span class="simple-quick-label">Duration</span>' +
                '<div id="simple-duration" class="simple-duration-row">' +
                    '<button class="simple-duration-btn" data-duration="short">Short (2s)</button>' +
                    '<button class="simple-duration-btn active" data-duration="medium">Medium (4s)</button>' +
                    '<button class="simple-duration-btn" data-duration="long">Long (8s)</button>' +
                '</div>' +
            '</div>' +
        '</div>' +

        // Generate button
        '<div class="simple-section">' +
            '<button id="simple-gen-btn" class="simple-gen-btn">Create Image</button>' +
        '</div>' +

        // Recent
        '<div class="simple-recent-section">' +
            '<div class="simple-recent-label">Recent</div>' +
            '<div id="simple-recent-grid" class="simple-recent-grid"></div>' +
        '</div>';
    }

    function buildCenterHTML() {
        return '' +
            '<div id="simple-empty" class="simple-empty">' +
                '<svg class="simple-empty-logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>' +
                '<span class="simple-empty-text">Your creation will appear here</span>' +
            '</div>' +
            '<img id="simple-preview-img" class="simple-preview-img" style="display:none" alt="Generated">' +
            '<video id="simple-preview-video" class="simple-preview-video" style="display:none" autoplay loop muted playsinline controls></video>' +
            '<div id="simple-action-bar" class="simple-action-bar" style="display:none">' +
                '<button class="simple-action-btn" id="simple-download" title="Download">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>' +
                '</button>' +
                '<button class="simple-action-btn" id="simple-variations" title="Variations (new seed)">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 18h1.4c1.3 0 2.5-.6 3.3-1.7l6.1-8.6c.7-1.1 2-1.7 3.3-1.7H22"/><path d="m18 2 4 4-4 4"/><path d="M2 6h1.9c1.5 0 2.9.9 3.6 2.2"/><path d="M22 18h-5.9c-1.3 0-2.6-.7-3.3-1.8l-.5-.8"/><path d="m18 14 4 4-4 4"/></svg>' +
                '</button>' +
                '<button class="simple-action-btn" id="simple-to-advanced" title="Send to Advanced">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>' +
                '</button>' +
                '<button class="simple-action-btn" id="simple-clear-preview" title="Clear">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>' +
                '</button>' +
            '</div>' +
            '<div id="simple-progress" class="simple-progress"><div id="simple-progress-bar" class="simple-progress-bar"></div></div>' +
            '<div id="simple-error-banner" class="simple-error-banner"></div>';
    }

    function cacheElements() {
        els.model = document.getElementById('simple-model');
        els.prompt = document.getElementById('simple-prompt');
        els.negPrompt = document.getElementById('simple-neg-prompt');
        els.advToggle = document.getElementById('simple-adv-toggle');
        els.advBody = document.getElementById('simple-adv-body');
        els.aspectGrid = document.getElementById('simple-aspects');
        els.qualityRow = document.getElementById('simple-quality');
        els.durationSection = document.getElementById('simple-duration-section');
        els.durationRow = document.getElementById('simple-duration');
        els.genBtn = document.getElementById('simple-gen-btn');
        els.recentGrid = document.getElementById('simple-recent-grid');
        els.empty = document.getElementById('simple-empty');
        els.previewImg = document.getElementById('simple-preview-img');
        els.previewVideo = document.getElementById('simple-preview-video');
        els.actionBar = document.getElementById('simple-action-bar');
        els.download = document.getElementById('simple-download');
        els.variations = document.getElementById('simple-variations');
        els.toAdvanced = document.getElementById('simple-to-advanced');
        els.clearPreview = document.getElementById('simple-clear-preview');
        els.progress = document.getElementById('simple-progress');
        els.progressBar = document.getElementById('simple-progress-bar');
        els.errorBanner = document.getElementById('simple-error-banner');
        els.stylesContainer = document.getElementById('simple-styles');
    }

    // ── Aspect Ratio Buttons ──

    function buildAspectButtons() {
        var aspects = getActiveAspects();
        els.aspectGrid.innerHTML = '';
        aspects.forEach(function(a) {
            var btn = document.createElement('button');
            var isActive = a.w === state.width && a.h === state.height;
            btn.className = 'simple-aspect-btn' + (isActive ? ' active' : '');
            var maxDim = 20;
            var sw = Math.round(a.vw / maxDim * 14);
            var sh = Math.round(a.vh / maxDim * 14);
            btn.innerHTML =
                '<svg width="' + (sw + 4) + '" height="' + (sh + 4) + '" viewBox="0 0 ' + (sw + 4) + ' ' + (sh + 4) + '" fill="none" stroke-width="1.5">' +
                    '<rect x="1" y="1" width="' + sw + '" height="' + sh + '" rx="2"/>' +
                '</svg>' +
                '<span class="simple-aspect-btn-label">' + a.label + '</span>';
            btn.addEventListener('click', function() {
                state.width = a.w;
                state.height = a.h;
                els.aspectGrid.querySelectorAll('.simple-aspect-btn').forEach(function(b) {
                    b.classList.remove('active');
                });
                btn.classList.add('active');
            });
            els.aspectGrid.appendChild(btn);
        });
    }

    // ── Event Binding ──

    function bindEvents() {
        // Prompt auto-grow
        els.prompt.addEventListener('input', function() {
            state.prompt = this.value;
            this.style.height = 'auto';
            this.style.height = Math.max(100, this.scrollHeight) + 'px';
        });

        els.negPrompt.addEventListener('input', function() {
            state.negPrompt = this.value;
        });

        // Advanced prompt disclosure
        els.advToggle.addEventListener('click', function() {
            this.classList.toggle('open');
            els.advBody.classList.toggle('open');
        });

        // Style presets
        els.stylesContainer.addEventListener('click', function(e) {
            var pill = e.target.closest('.simple-style-pill');
            if (!pill) return;
            var styleId = pill.dataset.style;
            state.activeStyle = styleId;
            els.stylesContainer.querySelectorAll('.simple-style-pill').forEach(function(p) {
                p.classList.toggle('active', p.dataset.style === styleId);
            });
        });

        // Quality presets
        els.qualityRow.addEventListener('click', function(e) {
            var btn = e.target.closest('.simple-quality-btn');
            if (!btn) return;
            var q = btn.dataset.quality;
            state.quality = q;
            state.steps = qualityPresets[q].steps;
            els.qualityRow.querySelectorAll('.simple-quality-btn').forEach(function(b) {
                b.classList.toggle('active', b.dataset.quality === q);
            });
        });

        // Duration presets
        els.durationRow.addEventListener('click', function(e) {
            var btn = e.target.closest('.simple-duration-btn');
            if (!btn) return;
            var d = btn.dataset.duration;
            state.duration = d;
            state.frames = durationPresets[d].frames;
            state.fps = durationPresets[d].fps;
            els.durationRow.querySelectorAll('.simple-duration-btn').forEach(function(b) {
                b.classList.toggle('active', b.dataset.duration === d);
            });
        });

        // Model change
        els.model.addEventListener('change', function() {
            state.model = this.value;
            updateUIForArch(ModelUtils.detectArchFromFilename(this.value));
            // Update topbar model badge
            var badge = document.querySelector('.model-badge');
            if (badge) badge.textContent = this.value;
        });

        // Generate
        els.genBtn.addEventListener('click', function() {
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

        els.variations.addEventListener('click', function() {
            if (state.generating || !state.prompt.trim()) return;
            state.seed = -1; // force new random seed
            generate();
        });

        els.toAdvanced.addEventListener('click', function() {
            if (typeof setMode === 'function') {
                // Store current image for Canvas tab to pick up
                if (state.currentImage) {
                    localStorage.setItem('sf-send-to-canvas', JSON.stringify({
                        src: state.currentImage,
                        isVideo: state.currentIsVideo
                    }));
                }
                setMode('advanced');
                switchTab('canvas');
            }
        });

        els.clearPreview.addEventListener('click', function() {
            clearPreview();
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
                // Update topbar badge
                var badge = document.querySelector('.model-badge');
                if (badge) badge.textContent = models[0].name;
            })
            .catch(function() {
                els.model.innerHTML = '<option disabled selected>No models found</option>';
            });
    }

    // ── Arch-aware UI ──

    function updateUIForArch(arch) {
        state.arch = arch;
        var isFlux = arch === 'flux';
        var isVideo = arch === 'ltxv' || arch === 'wan';

        // Auto-set CFG/Guidance based on arch
        if (isFlux || isVideo) {
            state.cfg = 1.0;
            state.guidance = 3.5;
        } else {
            state.cfg = 7.0;
        }

        // Video duration section
        els.durationSection.style.display = isVideo ? 'block' : 'none';

        // Button label
        els.genBtn.textContent = isVideo ? 'Create Video' : 'Create Image';

        // Rebuild aspect buttons
        buildAspectButtons();

        // Select first aspect
        var aspects = getActiveAspects();
        if (aspects.length > 0) {
            state.width = aspects[0].w;
            state.height = aspects[0].h;
        }

        // Apply current duration preset for video
        if (isVideo) {
            var dp = durationPresets[state.duration];
            state.frames = dp.frames;
            state.fps = dp.fps;
        }
    }

    // ── Workflow Builder ──

    function getEffectivePrompt() {
        var prompt = state.prompt;
        if (state.activeStyle && state.activeStyle !== 'none') {
            var style = stylePresets.find(function(s) { return s.id === state.activeStyle; });
            if (style && style.suffix) {
                prompt += style.suffix;
            }
        }
        return prompt;
    }

    function buildWorkflow() {
        return WorkflowBuilder.build({
            model: state.model,
            prompt: getEffectivePrompt(),
            negPrompt: state.negPrompt,
            width: state.width,
            height: state.height,
            steps: state.steps,
            cfg: state.cfg,
            guidance: state.guidance,
            scheduler: state.scheduler,
            seed: state.seed,
            frames: state.frames,
            fps: state.fps
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
        var workflow = buildWorkflow();

        fetch('/prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: workflow, client_id: SerenityWS.getClientId() })
        })
        .then(function(resp) {
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
        .catch(function(err) {
            showError('Failed to queue: ' + err.message);
            setGenerating(false);
        });
    }

    // ── State Helpers ──

    function setGenerating(v) {
        state.generating = v;
        els.genBtn.disabled = v;
        if (v) {
            els.genBtn.textContent = 'Creating...';
            els.progress.classList.add('active');
            els.progressBar.style.width = '100%';
        } else {
            els.genBtn.textContent = isVideoModel() ? 'Create Video' : 'Create Image';
            els.progress.classList.remove('active');
            els.progressBar.style.width = '0%';
        }
        els.genBtn.classList.toggle('generating', v);
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
        els.recentGrid.querySelectorAll('.simple-recent-thumb').forEach(function(t) {
            t.classList.remove('active');
        });
    }

    function addToRecent(src, isVideo) {
        state.recent.unshift({ src: src, isVideo: !!isVideo });
        if (state.recent.length > 6) state.recent.pop();
        renderRecent();
        saveRecent();
    }

    function renderRecent() {
        els.recentGrid.innerHTML = '';
        state.recent.forEach(function(item, i) {
            var thumb = document.createElement('div');
            thumb.className = 'simple-recent-thumb' + (i === 0 ? ' active' : '');
            if (item.isVideo) {
                thumb.innerHTML =
                    '<video src="' + item.src + '" muted preload="metadata"></video>' +
                    '<div class="simple-recent-play">\u25b6</div>';
            } else {
                thumb.innerHTML = '<img src="' + item.src + '" alt="recent">';
            }
            thumb.addEventListener('click', function() {
                if (item.isVideo) {
                    displayVideo(item.src);
                } else {
                    displayImage(item.src);
                }
                els.recentGrid.querySelectorAll('.simple-recent-thumb').forEach(function(t) {
                    t.classList.remove('active');
                });
                thumb.classList.add('active');
            });
            els.recentGrid.appendChild(thumb);
        });
    }

    function saveRecent() {
        try {
            localStorage.setItem('sf-simple-recent', JSON.stringify(state.recent));
        } catch(e) { /* quota */ }
    }

    function restoreRecent() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-simple-recent'));
            if (saved && Array.isArray(saved)) {
                state.recent = saved.slice(0, 6);
                renderRecent();
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

    // ── WebSocket ──

    function connectWS() {
        SerenityWS.on('progress', function(data) {
            if (!data || !state.generating) return;
            var pct = (data.value / data.max * 100).toFixed(0);
            els.progressBar.style.width = pct + '%';
        });

        SerenityWS.on('executed', function(data) {
            if (!data || !data.output) return;
            var items = data.output.images;
            var isVideo = false;
            if (!items && data.output.videos) {
                items = data.output.videos;
                isVideo = true;
            }
            if (!items || !items.length) return;

            var file = items[0];
            var src = '/view?filename=' + encodeURIComponent(file.filename) +
                '&subfolder=' + encodeURIComponent(file.subfolder || '') +
                '&type=' + encodeURIComponent(file.type || 'output');
            if (!isVideo) isVideo = /\.(webp|mp4|gif)$/i.test(file.filename);
            if (isVideo) {
                displayVideo(src);
            } else {
                displayImage(src);
            }
            addToRecent(src, isVideo);
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
        restoreRecent();
        connectWS();
    }

    return {
        state: state,
        init: init,
        generate: generate
    };
})();
