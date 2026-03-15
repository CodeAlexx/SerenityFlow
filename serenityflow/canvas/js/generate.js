/**
 * Generate Tab — SerenityFlow Phase 2
 * Prompt-to-image generation wired to ComfyUI-compatible backend.
 * InvokeAI-inspired visual overhaul.
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
        lockedRatio: 1,
        // New state for overhaul
        leftPanelVisible: true,
        rightPanelVisible: true,
        galleryTab: 'gallery',    // 'layers' | 'gallery'
        gallerySubTab: 'images',  // 'images' | 'assets'
        boardsVisible: true,
        selectedAspect: '1:1',
        pendingBatch: 0
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
        left.id = 'gen-left-panel';
        left.innerHTML = buildLeftHTML();
        layout.appendChild(left);

        // Center panel
        var center = document.createElement('div');
        center.className = 'gen-center';
        center.innerHTML = buildTopToolbarHTML() + buildCenterHTML();
        layout.appendChild(center);

        // Floating side toolbar (inside center so it positions relative to center)
        var floatBar = document.createElement('div');
        floatBar.className = 'gen-floating-toolbar';
        floatBar.id = 'gen-floating-toolbar';
        floatBar.innerHTML = buildFloatingToolbarHTML();
        center.appendChild(floatBar);

        // Right panel
        var right = document.createElement('div');
        right.className = 'gen-right';
        right.id = 'gen-right-panel';
        right.innerHTML = buildRightHTML();
        layout.appendChild(right);

        panel.appendChild(layout);
        cacheElements();
    }

    function buildLeftHTML() {
        return '' +
        // Generate button + progress bar at top
        '<div class="gen-section gen-top-actions">' +
            '<button id="gen-btn" class="gen-btn"><i data-lucide="wand-2"></i> Generate</button>' +
            '<div id="gen-left-progress" class="gen-progress gen-left-progress"><div id="gen-left-progress-bar" class="gen-progress-bar"></div></div>' +
            '<div id="gen-left-progress-label" class="gen-left-progress-label"></div>' +
        '</div>' +

        // Prompt
        '<div class="gen-section">' +
            '<div class="gen-prompt-label-row">' +
                '<label class="gen-label">Positive Prompt</label>' +
                '<div class="gen-prompt-label-actions">' +
                    '<button class="gen-prompt-label-btn" title="Dynamic prompts">' +
                        '&lt;/&gt;' +
                        '<span class="gen-tooltip">Dynamic prompts coming soon</span>' +
                    '</button>' +
                    '<button class="gen-prompt-label-btn" title="Templates">' +
                        '{}' +
                        '<span class="gen-tooltip">Templates coming soon</span>' +
                    '</button>' +
                '</div>' +
            '</div>' +
            '<textarea id="gen-prompt" class="gen-textarea" rows="4" placeholder="Describe your image..."></textarea>' +
            '<div id="gen-token-count" class="gen-token-count">~0 tokens</div>' +
        '</div>' +

        // Negative prompt
        '<div class="gen-section" id="gen-neg-section">' +
            '<label class="gen-label">Negative Prompt</label>' +
            '<textarea id="gen-neg-prompt" class="gen-textarea" rows="2" placeholder="What to avoid..."></textarea>' +
        '</div>' +

        // Image Settings accordion
        '<div class="gen-section">' +
            '<div id="gen-image-header" class="gen-accordion-header">' +
                '<span>Image</span>' +
                '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-image-body" class="gen-accordion-body" style="margin-top:8px">' +
                // Aspect row: dropdown + swap + lock + optimal
                '<div class="gen-image-content">' +
                    '<div class="gen-image-controls">' +
                        '<div class="gen-aspect-row">' +
                            '<span class="gen-label" style="margin-bottom:0;min-width:44px">Aspect</span>' +
                            '<select id="gen-aspect-dropdown" class="gen-select" style="flex:1">' +
                                '<option value="1:1">1:1</option>' +
                                '<option value="4:3">4:3</option>' +
                                '<option value="16:9">16:9</option>' +
                                '<option value="3:4">3:4</option>' +
                                '<option value="9:16">9:16</option>' +
                                '<option value="custom">Custom</option>' +
                            '</select>' +
                            '<button id="gen-swap-btn" class="gen-aspect-action" title="Swap width and height">' +
                                '<i data-lucide="arrow-left-right"></i>' +
                            '</button>' +
                            '<button id="gen-aspect-lock" class="gen-aspect-action' + (state.aspectLocked ? ' active' : '') + '" title="Lock aspect ratio">' +
                                '<i data-lucide="lock"></i>' +
                            '</button>' +
                            '<button id="gen-optimal-btn" class="gen-aspect-action" title="Set optimal size for model">' +
                                '<i data-lucide="sparkles"></i>' +
                            '</button>' +
                        '</div>' +
                        '<div class="gen-dim-row">' +
                            '<span class="gen-dim-label">Width</span>' +
                            '<input type="range" id="gen-width-slider" class="gen-range" min="256" max="2048" step="64" value="1024">' +
                            '<input type="number" id="gen-custom-width" class="gen-number-input" min="256" max="4096" step="64" value="1024">' +
                        '</div>' +
                        '<div class="gen-dim-row">' +
                            '<span class="gen-dim-label">Height</span>' +
                            '<input type="range" id="gen-height-slider" class="gen-range" min="256" max="2048" step="64" value="1024">' +
                            '<input type="number" id="gen-custom-height" class="gen-number-input" min="256" max="4096" step="64" value="1024">' +
                        '</div>' +
                        // Seed inside Image section
                        '<div style="margin-top:8px">' +
                            '<label class="gen-label">Seed</label>' +
                            '<div class="gen-seed-row">' +
                                '<input id="gen-seed" type="number" class="gen-number-input" value="-1">' +
                                '<button id="gen-seed-shuffle" class="gen-seed-btn" title="Shuffle seed">\u21BB</button>' +
                                '<button id="gen-seed-prev" class="gen-seed-btn" title="Use previous seed">\u21BA</button>' +
                                '<span class="gen-seed-random-label">Random</span>' +
                                '<button id="gen-seed-random-toggle" class="gen-toggle' + (state.seed === -1 ? ' on' : '') + '"></button>' +
                            '</div>' +
                        '</div>' +
                        // Advanced Options disclosure
                        '<div id="gen-image-adv-disclosure" class="gen-adv-disclosure">' +
                            '<i data-lucide="chevron-right"></i>' +
                            '<span>Advanced Options</span>' +
                        '</div>' +
                        '<div id="gen-image-adv-body" class="gen-adv-body"></div>' +
                    '</div>' +
                    // Visual aspect preview box
                    '<div class="gen-image-preview-col">' +
                        '<div id="gen-aspect-preview" class="gen-aspect-preview" style="width:60px;height:60px">' +
                            '<span>1:1</span>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
            '</div>' +
        '</div>' +

        // Generation Settings accordion
        '<div class="gen-section">' +
            '<div id="gen-settings-header" class="gen-accordion-header">' +
                '<div class="gen-section-header-row">' +
                    '<span>Generation</span>' +
                    '<span id="gen-model-badge" class="gen-model-badge" style="display:none"></span>' +
                    '<span id="gen-arch-badge" class="gen-arch-badge">SD1.5</span>' +
                '</div>' +
                '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-settings-body" class="gen-accordion-body" style="margin-top:8px">' +
                // Model
                '<label class="gen-label">Model</label>' +
                '<div class="gen-model-row">' +
                    '<select id="gen-model" class="gen-select"><option disabled selected>Loading models...</option></select>' +
                    '<button id="gen-model-refresh" class="gen-model-action-btn" title="Refresh models"><i data-lucide="refresh-cw"></i></button>' +
                    '<button class="gen-model-action-btn" title="Model settings (coming soon)"><i data-lucide="settings"></i></button>' +
                '</div>' +
                '<div id="gen-model-warn" class="gen-model-warning"></div>' +
                // Concepts / LoRA
                '<label class="gen-label" style="margin-top:8px">Concepts</label>' +
                '<div id="gen-lora-list" class="gen-lora-list"></div>' +
                '<select id="gen-lora-picker" class="gen-lora-dropdown">' +
                    '<option disabled selected>No LoRAs loaded</option>' +
                '</select>' +
                // Advanced Options for Generation (default OPEN)
                '<div id="gen-gen-adv-disclosure" class="gen-adv-disclosure open" style="margin-top:8px">' +
                    '<i data-lucide="chevron-right"></i>' +
                    '<span>Advanced Options</span>' +
                '</div>' +
                '<div id="gen-gen-adv-body" class="gen-adv-body open">' +
                    // Steps
                    '<div class="gen-setting-row" style="margin-top:6px">' +
                        '<span class="gen-label">Steps</span>' +
                        '<input id="gen-steps-range" type="range" class="gen-range" min="1" max="150" value="20">' +
                        '<input id="gen-steps" type="number" class="gen-number-input" min="1" max="150" value="20">' +
                    '</div>' +
                    // CFG
                    '<div id="gen-cfg-row" class="gen-setting-row">' +
                        '<span class="gen-label">CFG</span>' +
                        '<input id="gen-cfg-range" type="range" class="gen-range" min="1" max="20" step="0.5" value="7.0">' +
                        '<input id="gen-cfg" type="number" class="gen-number-input" min="1" max="20" step="0.5" value="7.0">' +
                    '</div>' +
                    // Guidance (FLUX only)
                    '<div id="gen-guidance-row" class="gen-setting-row" style="display:none">' +
                        '<span class="gen-label">Guidance</span>' +
                        '<input id="gen-guidance-range" type="range" class="gen-range" min="1" max="10" step="0.5" value="3.5">' +
                        '<input id="gen-guidance" type="number" class="gen-number-input" min="1" max="10" step="0.5" value="3.5">' +
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
                    // Batch
                    '<div class="gen-setting-row" id="gen-batch-section">' +
                        '<span class="gen-label">Batch</span>' +
                        '<input id="gen-batch" type="number" class="gen-number-input" min="1" max="8" value="1">' +
                        '<span class="gen-batch-hint">images per run</span>' +
                    '</div>' +
                '</div>' +
            '</div>' +
        '</div>' +

        // Video controls (hidden by default, shown for video models)
        '<div class="gen-section" id="gen-video-section" style="display:none">' +
            '<label class="gen-label">Video</label>' +
            '<div class="gen-setting-row">' +
                '<span class="gen-label" style="min-width:52px;margin-bottom:0">Frames</span>' +
                '<input type="range" id="gen-frames-range" class="gen-range" min="9" max="257" step="8" value="97">' +
                '<input type="number" id="gen-frames" class="gen-number-input" min="9" max="257" step="8" value="97">' +
            '</div>' +
            '<div class="gen-setting-row">' +
                '<span class="gen-label" style="min-width:52px;margin-bottom:0">FPS</span>' +
                '<input type="range" id="gen-fps-range" class="gen-range" min="8" max="60" step="1" value="24">' +
                '<input type="number" id="gen-fps" class="gen-number-input" min="8" max="60" step="1" value="24">' +
            '</div>' +
            '<div id="gen-duration-hint" class="gen-duration-hint"></div>' +
        '</div>' +

        // Compositing section (stub)
        '<div class="gen-section">' +
            '<div id="gen-compositing-header" class="gen-accordion-header closed">' +
                '<span>Compositing</span>' +
                '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-compositing-body" class="gen-accordion-body closed" style="margin-top:8px">' +
                '<div class="gen-stub-content">Coming soon</div>' +
            '</div>' +
        '</div>' +

        // Refiner section (stub)
        '<div class="gen-section">' +
            '<div id="gen-refiner-header" class="gen-accordion-header closed">' +
                '<span>Refiner</span>' +
                '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-refiner-body" class="gen-accordion-body closed" style="margin-top:8px">' +
                '<div class="gen-stub-content">SDXL Refiner settings</div>' +
            '</div>' +
        '</div>' +

        // Advanced section (stub)
        '<div class="gen-section">' +
            '<div id="gen-advanced-header" class="gen-accordion-header closed">' +
                '<span>Advanced</span>' +
                '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-advanced-body" class="gen-accordion-body closed" style="margin-top:8px">' +
                '<div class="gen-stub-control">' +
                    '<span class="gen-label">VAE</span>' +
                    '<select class="gen-select" style="flex:1" disabled><option>Default</option></select>' +
                '</div>' +
                '<div class="gen-stub-control">' +
                    '<span class="gen-label">Clip Skip</span>' +
                    '<input type="number" class="gen-number-input" value="0" disabled>' +
                '</div>' +
                '<div class="gen-stub-control">' +
                    '<span class="gen-label">CFG Rescale</span>' +
                    '<input type="number" class="gen-number-input" value="0" step="0.05" disabled>' +
                '</div>' +
                '<div class="gen-stub-control">' +
                    '<span class="gen-label">Seamless</span>' +
                    '<button class="gen-toggle" disabled></button>' +
                '</div>' +
            '</div>' +
        '</div>';
    }

    function buildTopToolbarHTML() {
        return '' +
        '<div class="gen-top-toolbar" id="gen-top-toolbar">' +
            // Generate button
            '<button id="gen-toolbar-generate" class="gen-toolbar-btn gen-toolbar-generate" title="Generate">' +
                '<i data-lucide="wand-2"></i>' +
                '<span>Invoke</span>' +
            '</button>' +
            // Batch count with spinners
            '<div class="gen-toolbar-batch">' +
                '<input type="number" id="gen-toolbar-batch-input" class="gen-toolbar-batch-input" min="1" max="8" value="1">' +
                '<div class="gen-toolbar-batch-spin">' +
                    '<button id="gen-toolbar-batch-up" title="Increase batch">\u25B2</button>' +
                    '<button id="gen-toolbar-batch-down" title="Decrease batch">\u25BC</button>' +
                '</div>' +
            '</div>' +
            '<span class="gen-toolbar-sep"></span>' +
            // View mode icons (stubs)
            '<button class="gen-toolbar-btn active" title="Image viewer"><i data-lucide="eye"></i></button>' +
            '<button class="gen-toolbar-btn" title="Brush mode"><i data-lucide="paintbrush"></i></button>' +
            '<button class="gen-toolbar-btn" title="List view"><i data-lucide="list"></i></button>' +
            '<button class="gen-toolbar-btn" title="Close view"><i data-lucide="x"></i></button>' +
            '<span class="gen-toolbar-sep"></span>' +
            // Center: info
            '<button class="gen-toolbar-btn" title="Image info"><i data-lucide="info"></i></button>' +
            '<span class="gen-toolbar-spacer"></span>' +
            // Toggle gallery panel
            '<button id="gen-toolbar-toggle-gallery" class="gen-toolbar-btn active" title="Toggle gallery"><i data-lucide="image"></i></button>' +
            '<span class="gen-toolbar-sep"></span>' +
            // Right group: undo, redo, flip-h, flip-v, star, copy, delete
            '<button class="gen-toolbar-btn" title="Undo" disabled><i data-lucide="undo-2"></i></button>' +
            '<button class="gen-toolbar-btn" title="Redo" disabled><i data-lucide="redo-2"></i></button>' +
            '<button class="gen-toolbar-btn" title="Flip horizontal" disabled><i data-lucide="flip-horizontal"></i></button>' +
            '<button class="gen-toolbar-btn" title="Flip vertical" disabled><i data-lucide="flip-vertical"></i></button>' +
            '<button class="gen-toolbar-btn" title="Favorite" disabled><i data-lucide="star"></i></button>' +
            '<button id="gen-toolbar-copy" class="gen-toolbar-btn" title="Copy image URL"><i data-lucide="copy"></i></button>' +
            '<button id="gen-toolbar-delete" class="gen-toolbar-btn" title="Delete"><i data-lucide="trash-2"></i></button>' +
        '</div>';
    }

    function buildFloatingToolbarHTML() {
        return '' +
            '<button id="gen-float-toggle-left" class="gen-float-btn" title="Toggle left panel"><i data-lucide="panel-left"></i></button>' +
            '<button id="gen-float-generate" class="gen-float-btn gen-float-generate" title="Generate"><i data-lucide="sparkles"></i></button>' +
            '<button id="gen-float-cancel" class="gen-float-btn gen-float-cancel" title="Cancel"><i data-lucide="x"></i></button>' +
            '<button id="gen-float-delete" class="gen-float-btn gen-float-delete" title="Delete current"><i data-lucide="trash-2"></i></button>';
    }

    function buildCenterHTML() {
        return '' +
            '<div class="gen-center-content">' +
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
            '</div>' +
            '<div id="gen-progress-label" class="gen-progress-label"></div>' +
            '<div id="gen-progress" class="gen-progress"><div id="gen-progress-bar" class="gen-progress-bar"></div></div>' +
            '<div id="gen-error-banner" class="gen-error-banner"></div>' +
            '<div id="gen-ws-indicator" class="gen-ws-indicator"><span class="gen-ws-dot"></span><span>Reconnecting...</span></div>';
    }

    function buildRightHTML() {
        return '' +
            // Tab switcher: Layers | Gallery | close
            '<div class="gen-gallery-tabs">' +
                '<button id="gen-tab-layers" class="gen-gallery-tab">Layers</button>' +
                '<button id="gen-tab-gallery" class="gen-gallery-tab active">Gallery</button>' +
                '<button id="gen-gallery-close" class="gen-gallery-close-btn" title="Close panel"><i data-lucide="x"></i></button>' +
            '</div>' +
            // Layers tab content
            '<div id="gen-layers-content" class="gen-gallery-tab-content">' +
                '<div class="gen-layers-placeholder">Coming soon</div>' +
            '</div>' +
            // Gallery tab content
            '<div id="gen-gallery-content" class="gen-gallery-tab-content active">' +
                // Boards section
                '<div class="gen-boards-section">' +
                    '<div id="gen-boards-header" class="gen-boards-header">' +
                        '<span>Hide Boards</span>' +
                        '<span class="gen-accordion-arrow"><i data-lucide="chevron-up"></i></span>' +
                    '</div>' +
                    '<div id="gen-boards-body" class="gen-boards-body">' +
                        '<div class="gen-board-item">' +
                            '<span><span class="gen-board-icon">\uD83D\uDCC1</span> Uncategorized</span>' +
                            '<span class="gen-board-badge">AUTO</span>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
                // Sub-tabs: Images | Assets | actions
                '<div class="gen-gallery-subtabs">' +
                    '<button id="gen-subtab-images" class="gen-gallery-subtab active">Images</button>' +
                    '<button id="gen-subtab-assets" class="gen-gallery-subtab">Assets</button>' +
                    '<div class="gen-gallery-subtab-actions">' +
                        '<button class="gen-gallery-subtab-btn" title="Upload (coming soon)"><i data-lucide="upload"></i></button>' +
                        '<button class="gen-gallery-subtab-btn" title="Settings (coming soon)"><i data-lucide="settings"></i></button>' +
                        '<button id="gen-gallery-search-btn" class="gen-gallery-subtab-btn" title="Search"><i data-lucide="search"></i></button>' +
                        '<input id="gen-gallery-search-input" class="gen-gallery-search" type="text" placeholder="Search...">' +
                    '</div>' +
                '</div>' +
                // Images sub-tab
                '<div id="gen-images-content">' +
                    '<div class="gen-gallery-header">' +
                        '<span class="gen-gallery-title">Gallery</span>' +
                        '<button id="gen-gallery-clear" class="gen-gallery-clear">Clear</button>' +
                    '</div>' +
                    '<div id="gen-gallery-grid" class="gen-gallery-grid"></div>' +
                '</div>' +
                // Assets sub-tab
                '<div id="gen-assets-content" style="display:none">' +
                    '<div class="gen-assets-placeholder">No assets uploaded</div>' +
                '</div>' +
                // Pagination
                '<div class="gen-gallery-pagination">' +
                    '<span>Page 1</span>' +
                '</div>' +
            '</div>';
    }

    function cacheElements() {
        els.model = document.getElementById('gen-model');
        els.modelWarn = document.getElementById('gen-model-warn');
        els.prompt = document.getElementById('gen-prompt');
        els.negPrompt = document.getElementById('gen-neg-prompt');
        els.negSection = document.getElementById('gen-neg-section');
        els.customWidth = document.getElementById('gen-custom-width');
        els.customHeight = document.getElementById('gen-custom-height');
        els.videoSection = document.getElementById('gen-video-section');
        els.framesInput = document.getElementById('gen-frames');
        els.framesRange = document.getElementById('gen-frames-range');
        els.fpsInput = document.getElementById('gen-fps');
        els.fpsRange = document.getElementById('gen-fps-range');
        els.durationHint = document.getElementById('gen-duration-hint');
        els.imageHeader = document.getElementById('gen-image-header');
        els.imageBody = document.getElementById('gen-image-body');
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
        els.leftProgress = document.getElementById('gen-left-progress');
        els.leftProgressBar = document.getElementById('gen-left-progress-bar');
        els.leftProgressLabel = document.getElementById('gen-left-progress-label');
        els.errorBanner = document.getElementById('gen-error-banner');
        els.wsIndicator = document.getElementById('gen-ws-indicator');
        els.galleryGrid = document.getElementById('gen-gallery-grid');
        els.galleryClear = document.getElementById('gen-gallery-clear');
        // New elements
        els.leftPanel = document.getElementById('gen-left-panel');
        els.rightPanel = document.getElementById('gen-right-panel');
        els.aspectDropdown = document.getElementById('gen-aspect-dropdown');
        els.aspectPreview = document.getElementById('gen-aspect-preview');
        els.modelBadge = document.getElementById('gen-model-badge');
        els.toolbarBatchInput = document.getElementById('gen-toolbar-batch-input');
    }

    // ── Aspect Preview Box ──
    function updateAspectPreview() {
        if (!els.aspectPreview) return;
        var maxDim = 60;
        var w = state.width;
        var h = state.height;
        var scale;
        if (w >= h) {
            scale = maxDim / w;
        } else {
            scale = maxDim / h;
        }
        var pw = Math.max(16, Math.round(w * scale));
        var ph = Math.max(16, Math.round(h * scale));
        els.aspectPreview.style.width = pw + 'px';
        els.aspectPreview.style.height = ph + 'px';
        // Determine label
        var label = state.width + '\u00d7' + state.height;
        var aspects = getActiveAspects();
        for (var i = 0; i < aspects.length; i++) {
            if (aspects[i].w === state.width && aspects[i].h === state.height) {
                label = aspects[i].label;
                break;
            }
        }
        els.aspectPreview.innerHTML = '<span>' + label + '</span>';
    }

    // ── Aspect Dropdown Sync ──
    function syncAspectDropdown() {
        if (!els.aspectDropdown) return;
        var aspects = getActiveAspects();
        var found = false;
        for (var i = 0; i < aspects.length; i++) {
            if (aspects[i].w === state.width && aspects[i].h === state.height) {
                els.aspectDropdown.value = aspects[i].label;
                found = true;
                break;
            }
        }
        if (!found) {
            els.aspectDropdown.value = 'custom';
        }
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

        // Image accordion
        els.imageHeader.addEventListener('click', function() {
            this.classList.toggle('closed');
            els.imageBody.classList.toggle('closed');
        });

        // Generation Settings accordion
        els.settingsHeader.addEventListener('click', function() {
            this.classList.toggle('closed');
            els.settingsBody.classList.toggle('closed');
        });

        // Stub accordions
        bindAccordion('gen-compositing-header', 'gen-compositing-body');
        bindAccordion('gen-refiner-header', 'gen-refiner-body');
        bindAccordion('gen-advanced-header', 'gen-advanced-body');

        // Image Advanced Options disclosure
        var imgAdvDisc = document.getElementById('gen-image-adv-disclosure');
        var imgAdvBody = document.getElementById('gen-image-adv-body');
        if (imgAdvDisc && imgAdvBody) {
            imgAdvDisc.addEventListener('click', function() {
                this.classList.toggle('open');
                imgAdvBody.classList.toggle('open');
            });
        }

        // Generation Advanced Options disclosure
        var genAdvDisc = document.getElementById('gen-gen-adv-disclosure');
        var genAdvBody = document.getElementById('gen-gen-adv-body');
        if (genAdvDisc && genAdvBody) {
            genAdvDisc.addEventListener('click', function() {
                this.classList.toggle('open');
                genAdvBody.classList.toggle('open');
            });
        }

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

        // Model refresh button
        var modelRefresh = document.getElementById('gen-model-refresh');
        if (modelRefresh) {
            modelRefresh.addEventListener('click', function(e) {
                e.stopPropagation();
                loadModels();
            });
        }

        // Guidance sync (FLUX)
        els.guidance.addEventListener('input', function() {
            state.guidance = parseFloat(this.value) || 3.5;
            els.guidanceRange.value = this.value;
        });
        els.guidanceRange.addEventListener('input', function() {
            state.guidance = parseFloat(this.value);
            els.guidance.value = this.value;
        });

        // Aspect dropdown
        if (els.aspectDropdown) {
            els.aspectDropdown.addEventListener('change', function() {
                var val = this.value;
                if (val === 'custom') return;
                var aspects = getActiveAspects();
                for (var i = 0; i < aspects.length; i++) {
                    if (aspects[i].label === val) {
                        state.width = aspects[i].w;
                        state.height = aspects[i].h;
                        syncDimensionInputs();
                        updateAspectPreview();
                        break;
                    }
                }
            });
        }

        // Swap button
        var swapBtn = document.getElementById('gen-swap-btn');
        if (swapBtn) {
            swapBtn.addEventListener('click', function() {
                var tmp = state.width;
                state.width = state.height;
                state.height = tmp;
                syncDimensionInputs();
                syncAspectDropdown();
                updateAspectPreview();
            });
        }

        // Aspect lock
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

        // Optimal size button
        var optBtn = document.getElementById('gen-optimal-btn');
        if (optBtn) {
            optBtn.addEventListener('click', function() {
                var defaults = getDefaultsForArch(state.arch);
                state.width = defaults.w;
                state.height = defaults.h;
                syncDimensionInputs();
                syncAspectDropdown();
                updateAspectPreview();
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
            var wsl = document.getElementById('gen-width-slider');
            if (wsl) wsl.value = v;
            if (state.aspectLocked && state.lockedRatio) {
                var newH = isVideo
                    ? ModelUtils.clampVideoDimension(Math.round(v / state.lockedRatio))
                    : ModelUtils.clampDimension(Math.round(v / state.lockedRatio));
                state.height = newH;
                els.customHeight.value = newH;
                var hsl = document.getElementById('gen-height-slider');
                if (hsl) hsl.value = newH;
            }
            syncAspectDropdown();
            updateAspectPreview();
        });
        els.customHeight.addEventListener('blur', function() {
            var isVideo = ModelUtils.isVideoModel(state.model);
            var v = isVideo
                ? ModelUtils.clampVideoDimension(parseInt(this.value) || 512)
                : ModelUtils.clampDimension(parseInt(this.value) || 1024);
            this.value = v;
            state.height = v;
            var hsl = document.getElementById('gen-height-slider');
            if (hsl) hsl.value = v;
            if (state.aspectLocked && state.lockedRatio) {
                var newW = isVideo
                    ? ModelUtils.clampVideoDimension(Math.round(v * state.lockedRatio))
                    : ModelUtils.clampDimension(Math.round(v * state.lockedRatio));
                state.width = newW;
                els.customWidth.value = newW;
                var wsl = document.getElementById('gen-width-slider');
                if (wsl) wsl.value = newW;
            }
            syncAspectDropdown();
            updateAspectPreview();
        });

        // Width slider sync
        var widthSlider = document.getElementById('gen-width-slider');
        if (widthSlider) {
            widthSlider.addEventListener('input', function() {
                var v = parseInt(this.value);
                state.width = v;
                els.customWidth.value = v;
                if (state.aspectLocked && state.lockedRatio) {
                    var newH = ModelUtils.clampDimension(Math.round(v / state.lockedRatio));
                    state.height = newH;
                    els.customHeight.value = newH;
                    var hs = document.getElementById('gen-height-slider');
                    if (hs) hs.value = newH;
                }
                syncAspectDropdown();
                updateAspectPreview();
            });
        }
        var heightSlider = document.getElementById('gen-height-slider');
        if (heightSlider) {
            heightSlider.addEventListener('input', function() {
                var v = parseInt(this.value);
                state.height = v;
                els.customHeight.value = v;
                if (state.aspectLocked && state.lockedRatio) {
                    var newW = ModelUtils.clampDimension(Math.round(v * state.lockedRatio));
                    state.width = newW;
                    els.customWidth.value = newW;
                    var ws = document.getElementById('gen-width-slider');
                    if (ws) ws.value = newW;
                }
                syncAspectDropdown();
                updateAspectPreview();
            });
        }

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

        // Seed random toggle
        var randomToggle = document.getElementById('gen-seed-random-toggle');
        if (randomToggle) {
            randomToggle.addEventListener('click', function() {
                var isRandom = state.seed !== -1;
                if (isRandom) {
                    state.seed = -1;
                    els.seed.value = -1;
                } else {
                    state.seed = Math.floor(Math.random() * 4294967296);
                    els.seed.value = state.seed;
                }
                this.classList.toggle('on', state.seed === -1);
            });
        }

        // Batch count (left panel)
        var batchInput = document.getElementById('gen-batch');
        if (batchInput) {
            batchInput.addEventListener('input', function() {
                state.batchCount = Math.max(1, Math.min(8, parseInt(this.value) || 1));
                if (els.toolbarBatchInput) els.toolbarBatchInput.value = state.batchCount;
            });
        }

        // Batch count (toolbar)
        if (els.toolbarBatchInput) {
            els.toolbarBatchInput.addEventListener('input', function() {
                state.batchCount = Math.max(1, Math.min(8, parseInt(this.value) || 1));
                if (batchInput) batchInput.value = state.batchCount;
            });
        }
        var batchUp = document.getElementById('gen-toolbar-batch-up');
        var batchDown = document.getElementById('gen-toolbar-batch-down');
        if (batchUp) {
            batchUp.addEventListener('click', function() {
                state.batchCount = Math.min(8, state.batchCount + 1);
                if (els.toolbarBatchInput) els.toolbarBatchInput.value = state.batchCount;
                if (batchInput) batchInput.value = state.batchCount;
            });
        }
        if (batchDown) {
            batchDown.addEventListener('click', function() {
                state.batchCount = Math.max(1, state.batchCount - 1);
                if (els.toolbarBatchInput) els.toolbarBatchInput.value = state.batchCount;
                if (batchInput) batchInput.value = state.batchCount;
            });
        }

        // LoRA picker (concepts dropdown)
        var loraPicker = document.getElementById('gen-lora-picker');
        if (loraPicker) {
            loraPicker.addEventListener('change', function() {
                if (this.value && this.selectedIndex > 0) {
                    addLora(this.value);
                    this.selectedIndex = 0;
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

        // Generate (left panel button)
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

        // ── Top Toolbar ──
        var toolbarGenerate = document.getElementById('gen-toolbar-generate');
        if (toolbarGenerate) {
            toolbarGenerate.addEventListener('click', function() { generate(); });
        }
        var toolbarDelete = document.getElementById('gen-toolbar-delete');
        if (toolbarDelete) {
            toolbarDelete.addEventListener('click', function() { clearPreview(); });
        }
        var toolbarToggleGallery = document.getElementById('gen-toolbar-toggle-gallery');
        if (toolbarToggleGallery) {
            toolbarToggleGallery.addEventListener('click', function() {
                state.rightPanelVisible = !state.rightPanelVisible;
                els.rightPanel.classList.toggle('gen-panel-hidden', !state.rightPanelVisible);
                this.classList.toggle('active', state.rightPanelVisible);
            });
        }
        var toolbarCopy = document.getElementById('gen-toolbar-copy');
        if (toolbarCopy) {
            toolbarCopy.addEventListener('click', function() {
                if (state.currentImage) {
                    var fullUrl = window.location.origin + state.currentImage;
                    navigator.clipboard.writeText(fullUrl).catch(function() {});
                }
            });
        }

        // ── Floating Toolbar ──
        var floatToggleLeft = document.getElementById('gen-float-toggle-left');
        if (floatToggleLeft) {
            floatToggleLeft.addEventListener('click', function() {
                state.leftPanelVisible = !state.leftPanelVisible;
                els.leftPanel.classList.toggle('gen-panel-hidden', !state.leftPanelVisible);
            });
        }
        var floatGenerate = document.getElementById('gen-float-generate');
        if (floatGenerate) {
            floatGenerate.addEventListener('click', function() { generate(); });
        }
        var floatCancel = document.getElementById('gen-float-cancel');
        if (floatCancel) {
            floatCancel.addEventListener('click', function() {
                if (state.generating) {
                    state.pendingBatch = 0;
                    SerenityAPI.interrupt();
                    setGenerating(false);
                }
            });
        }
        var floatDelete = document.getElementById('gen-float-delete');
        if (floatDelete) {
            floatDelete.addEventListener('click', function() { clearPreview(); });
        }

        // ── Gallery Tabs ──
        var tabLayers = document.getElementById('gen-tab-layers');
        var tabGallery = document.getElementById('gen-tab-gallery');
        var layersContent = document.getElementById('gen-layers-content');
        var galleryContent = document.getElementById('gen-gallery-content');
        if (tabLayers && tabGallery) {
            tabLayers.addEventListener('click', function() {
                state.galleryTab = 'layers';
                tabLayers.classList.add('active');
                tabGallery.classList.remove('active');
                layersContent.classList.add('active');
                galleryContent.classList.remove('active');
            });
            tabGallery.addEventListener('click', function() {
                state.galleryTab = 'gallery';
                tabGallery.classList.add('active');
                tabLayers.classList.remove('active');
                galleryContent.classList.add('active');
                layersContent.classList.remove('active');
            });
        }

        // Gallery close
        var galleryClose = document.getElementById('gen-gallery-close');
        if (galleryClose) {
            galleryClose.addEventListener('click', function() {
                state.rightPanelVisible = false;
                els.rightPanel.classList.add('gen-panel-hidden');
            });
        }

        // Boards toggle
        var boardsHeader = document.getElementById('gen-boards-header');
        var boardsBody = document.getElementById('gen-boards-body');
        if (boardsHeader && boardsBody) {
            boardsHeader.addEventListener('click', function() {
                state.boardsVisible = !state.boardsVisible;
                boardsBody.classList.toggle('closed', !state.boardsVisible);
                boardsHeader.classList.toggle('closed', !state.boardsVisible);
                boardsHeader.querySelector('span:first-child').textContent = state.boardsVisible ? 'Hide Boards' : 'Show Boards';
            });
        }

        // Sub-tabs
        var subtabImages = document.getElementById('gen-subtab-images');
        var subtabAssets = document.getElementById('gen-subtab-assets');
        var imagesContent = document.getElementById('gen-images-content');
        var assetsContent = document.getElementById('gen-assets-content');
        if (subtabImages && subtabAssets) {
            subtabImages.addEventListener('click', function() {
                state.gallerySubTab = 'images';
                subtabImages.classList.add('active');
                subtabAssets.classList.remove('active');
                imagesContent.style.display = '';
                assetsContent.style.display = 'none';
            });
            subtabAssets.addEventListener('click', function() {
                state.gallerySubTab = 'assets';
                subtabAssets.classList.add('active');
                subtabImages.classList.remove('active');
                assetsContent.style.display = '';
                imagesContent.style.display = 'none';
            });
        }

        // Gallery search toggle
        var searchBtn = document.getElementById('gen-gallery-search-btn');
        var searchInput = document.getElementById('gen-gallery-search-input');
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', function() {
                searchInput.classList.toggle('open');
                if (searchInput.classList.contains('open')) {
                    searchInput.focus();
                } else {
                    searchInput.value = '';
                }
            });
        }
    }

    function bindAccordion(headerId, bodyId) {
        var header = document.getElementById(headerId);
        var body = document.getElementById(bodyId);
        if (header && body) {
            header.addEventListener('click', function() {
                this.classList.toggle('closed');
                body.classList.toggle('closed');
            });
        }
    }

    function syncDimensionInputs() {
        els.customWidth.value = state.width;
        els.customHeight.value = state.height;
        var ws = document.getElementById('gen-width-slider');
        var hs = document.getElementById('gen-height-slider');
        if (ws) ws.value = state.width;
        if (hs) hs.value = state.height;
    }

    function getDefaultsForArch(arch) {
        var defaults = {
            sd15:  { w: 512,  h: 512 },
            sdxl:  { w: 1024, h: 1024 },
            sd3:   { w: 1024, h: 1024 },
            flux:  { w: 1024, h: 1024 },
            klein: { w: 1024, h: 1024 },
            ltxv:  { w: 512,  h: 512 },
            wan:   { w: 768,  h: 432 }
        };
        return defaults[arch] || { w: 1024, h: 1024 };
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
        var isFlux = arch === 'flux' || arch === 'klein';
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
        var batchSection = document.getElementById('gen-batch-section');
        if (batchSection) batchSection.style.display = isVideo ? 'none' : 'flex';

        // Button label
        els.btn.innerHTML = isVideo
            ? '<i data-lucide="wand-2"></i> Generate Video'
            : '<i data-lucide="wand-2"></i> Generate';
        if (typeof lucide !== 'undefined') lucide.createIcons({nameAttr: 'data-lucide'});

        // Update arch badge
        var archBadge = document.getElementById('gen-arch-badge');
        if (archBadge) {
            var archNames = { flux: 'FLUX', sdxl: 'SDXL', sd3: 'SD3', sd15: 'SD1.5', ltxv: 'LTX-V', wan: 'WAN', klein: 'KLEIN' };
            archBadge.textContent = archNames[arch] || arch.toUpperCase();
            archBadge.dataset.arch = arch;
        }

        // Update model name badge
        if (els.modelBadge && state.model) {
            var shortName = state.model.length > 20 ? state.model.substring(0, 18) + '...' : state.model;
            els.modelBadge.textContent = shortName;
            els.modelBadge.style.display = '';
        }

        // Update custom res input constraints
        var ws = document.getElementById('gen-width-slider');
        var hs = document.getElementById('gen-height-slider');
        if (isVideo) {
            els.customWidth.min = 64;
            els.customWidth.max = 1280;
            els.customWidth.step = 32;
            els.customHeight.min = 64;
            els.customHeight.max = 1280;
            els.customHeight.step = 32;
            if (ws) { ws.min = 64; ws.max = 1280; ws.step = 32; }
            if (hs) { hs.min = 64; hs.max = 1280; hs.step = 32; }
        } else {
            els.customWidth.min = 256;
            els.customWidth.max = 4096;
            els.customWidth.step = 64;
            els.customHeight.min = 256;
            els.customHeight.max = 4096;
            els.customHeight.step = 64;
            if (ws) { ws.min = 256; ws.max = 2048; ws.step = 64; }
            if (hs) { hs.min = 256; hs.max = 2048; hs.step = 64; }
        }

        // Update aspect dropdown options for video vs image
        if (els.aspectDropdown) {
            var aspects = getActiveAspects();
            els.aspectDropdown.innerHTML = '';
            aspects.forEach(function(a) {
                var opt = document.createElement('option');
                opt.value = a.label;
                opt.textContent = a.label;
                els.aspectDropdown.appendChild(opt);
            });
            var customOpt = document.createElement('option');
            customOpt.value = 'custom';
            customOpt.textContent = 'Custom';
            els.aspectDropdown.appendChild(customOpt);
        }

        // Select first aspect ratio for the new mode
        var aspects = getActiveAspects();
        if (aspects.length > 0) {
            state.width = aspects[0].w;
            state.height = aspects[0].h;
            syncDimensionInputs();
            syncAspectDropdown();
            updateAspectPreview();
        }

        if (isVideo) updateDurationHint();
    }

    function updateDurationHint() {
        if (!els.durationHint) return;
        var secs = (state.frames / state.fps).toFixed(1);
        els.durationHint.textContent = '\u2248 ' + secs + 's at ' + state.fps + 'fps';
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
                picker.innerHTML = '<option disabled selected>No LoRAs loaded</option>';
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

        var batchN = state.batchCount || 1;
        state.pendingBatch = batchN;
        setGenerating(true);
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
                els.btn.innerHTML = '<i data-lucide="wand-2"></i> Generating ' + (i + 1) + ' / ' + batchN + '...';
                if (typeof lucide !== 'undefined') lucide.createIcons({nameAttr: 'data-lucide'});
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
            els.btn.innerHTML = '<i data-lucide="wand-2"></i> Generating...';
        } else {
            els.btn.innerHTML = isVideo
                ? '<i data-lucide="wand-2"></i> Generate Video'
                : '<i data-lucide="wand-2"></i> Generate';
        }
        if (typeof lucide !== 'undefined') lucide.createIcons({nameAttr: 'data-lucide'});
        els.btn.classList.toggle('generating', v);
        if (v) {
            els.progress.classList.add('active');
            els.progressBar.style.width = '100%';
            if (els.leftProgress) {
                els.leftProgress.classList.add('active');
                els.leftProgressBar.style.width = '100%';
            }
        } else {
            els.progress.classList.remove('active');
            els.progressBar.style.width = '0%';
            els.progressLabel.classList.remove('visible');
            if (els.leftProgress) {
                els.leftProgress.classList.remove('active');
                els.leftProgressBar.style.width = '0%';
            }
            if (els.leftProgressLabel) {
                els.leftProgressLabel.textContent = '';
            }
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
        state.gallery.unshift({ src: src, isVideo: !!isVideo, prompt: state.prompt });
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
            if (els.leftProgressBar) {
                els.leftProgressBar.style.width = pct + '%';
            }
            if (els.leftProgressLabel) {
                els.leftProgressLabel.textContent = 'Step ' + data.value + ' / ' + data.max;
            }
        });

        SerenityWS.on('preview', function(data) {
            if (!data || !data.blob || !state.generating) return;
            var url = URL.createObjectURL(data.blob);
            if (els.previewImg) {
                // Revoke old preview URL to prevent memory leaks
                if (els.previewImg._previewUrl) URL.revokeObjectURL(els.previewImg._previewUrl);
                els.previewImg._previewUrl = url;
                els.previewImg.src = url;
                els.previewImg.style.display = 'block';
                if (els.empty) els.empty.style.display = 'none';
                els.previewImg.classList.add('gen-preview-live');
            }
        });

        SerenityWS.on('executed', function(data) {
            if (!data || !data.output) return;
            // Clean up live preview state
            if (els.previewImg) {
                els.previewImg.classList.remove('gen-preview-live');
                if (els.previewImg._previewUrl) {
                    URL.revokeObjectURL(els.previewImg._previewUrl);
                    els.previewImg._previewUrl = null;
                }
            }
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
            state.pendingBatch = Math.max(0, state.pendingBatch - 1);
            if (state.pendingBatch <= 0) {
                setGenerating(false);
            }
        });

        SerenityWS.on('execution_error', function(data) {
            state.pendingBatch = 0;
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
        bindEvents();
        loadModels();
        loadLoras();
        restoreGallery();
        connectWS();
        updateAspectPreview();

        // Render lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons({nameAttr: 'data-lucide'});
        }
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
