/**
 * Settings Tab — SerenityFlow Phase 9
 * Persistent user preferences with theme, memory, output, and interface controls.
 */

// ── Settings Persistence Layer (loaded before any tab) ──
var Settings = (function() {
    'use strict';

    var DEFAULTS = {
        vramBudgetMb: 0,
        cpuOffloadEnabled: true,
        pinned_pool_mb: 512,
        outputDir: '',
        outputFormat: 'png',
        jpegQuality: 95,
        saveMetadata: true,
        theme: 'dark',
        accentColor: '#6c6af5',
        defaultMode: 'simple',
        defaultTab: 'generate',
        galleryColumns: 2,
        showNodeIds: false,
        confirmOnClear: true,
        defaultModel: '',
        defaultSteps: 20,
        defaultCfg: 7.0,
        defaultScheduler: 'euler'
    };

    var current = {};

    function load() {
        try {
            var stored = JSON.parse(localStorage.getItem('sf-settings') || '{}');
            current = merge(DEFAULTS, stored);
        } catch(e) {
            current = merge(DEFAULTS, {});
        }
        return current;
    }

    function merge(defaults, overrides) {
        var result = {};
        for (var k in defaults) {
            result[k] = overrides.hasOwnProperty(k) ? overrides[k] : defaults[k];
        }
        return result;
    }

    function save() {
        try {
            localStorage.setItem('sf-settings', JSON.stringify(current));
        } catch(e) {}
    }

    function get(key) { return current[key]; }

    function set(key, value) {
        current[key] = value;
        save();
        applyImmediate(key, value);
    }

    function getAll() {
        var copy = {};
        for (var k in current) copy[k] = current[k];
        return copy;
    }

    function reset() {
        current = merge(DEFAULTS, {});
        save();
        location.reload();
    }

    function applyImmediate(key, value) {
        if (key === 'accentColor') applyAccentColor(value);
        if (key === 'theme') applyTheme(value);
    }

    // ── Theme ──
    function applyTheme(theme) {
        var root = document.documentElement;
        var isDark = theme === 'dark' ||
            (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

        if (isDark) {
            root.style.setProperty('--shell-bg-base', '#0f0f13');
            root.style.setProperty('--shell-bg-surface', '#1a1a24');
            root.style.setProperty('--shell-bg-panel', '#1f1f2e');
            root.style.setProperty('--shell-bg-hover', '#252538');
            root.style.setProperty('--shell-text', '#e8e8f0');
            root.style.setProperty('--shell-text-muted', '#6b6b80');
            root.style.setProperty('--shell-border', '#2a2a3a');
        } else {
            root.style.setProperty('--shell-bg-base', '#f4f4f8');
            root.style.setProperty('--shell-bg-surface', '#ffffff');
            root.style.setProperty('--shell-bg-panel', '#f0f0f5');
            root.style.setProperty('--shell-bg-hover', '#e8e8ef');
            root.style.setProperty('--shell-text', '#1a1a2e');
            root.style.setProperty('--shell-text-muted', '#6b6b80');
            root.style.setProperty('--shell-border', '#d0d0de');
        }
    }

    function applyAccentColor(hex) {
        if (!/^#[0-9a-fA-F]{6}$/.test(hex)) return;
        document.documentElement.style.setProperty('--shell-accent', hex);
        // Derive hover: lighten by 15%
        var num = parseInt(hex.slice(1), 16);
        var r = Math.min(255, (num >> 16) + 38);
        var g = Math.min(255, ((num >> 8) & 0xff) + 38);
        var b = Math.min(255, (num & 0xff) + 38);
        var hover = '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
        document.documentElement.style.setProperty('--shell-accent-hover', hover);
    }

    // System theme listener
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
        if (current.theme === 'system') applyTheme('system');
    });

    return {
        load: load, save: save, get: get, set: set,
        getAll: getAll, reset: reset, DEFAULTS: DEFAULTS,
        applyTheme: applyTheme, applyAccentColor: applyAccentColor
    };
})();

// Load immediately
Settings.load();
Settings.applyTheme(Settings.get('theme'));
Settings.applyAccentColor(Settings.get('accentColor'));


// ── Settings Tab UI ──
var SettingsTab = (function() {
    'use strict';

    var initialized = false;
    var activeSection = 'general';

    var ACCENT_PRESETS = [
        { hex: '#6c6af5', label: 'Violet' },
        { hex: '#3b82f6', label: 'Blue' },
        { hex: '#14b8a6', label: 'Teal' },
        { hex: '#22c55e', label: 'Green' },
        { hex: '#f43f5e', label: 'Rose' },
        { hex: '#f59e0b', label: 'Amber' }
    ];

    function buildUI() {
        var panel = document.getElementById('panel-settings');
        if (!panel) return;
        panel.innerHTML = '';

        var layout = document.createElement('div');
        layout.className = 'settings-layout';
        layout.innerHTML =
            '<div class="settings-nav">' +
                '<div class="settings-nav-item active" data-section="general">General</div>' +
                '<div class="settings-nav-item" data-section="memory">Memory</div>' +
                '<div class="settings-nav-item" data-section="output">Output</div>' +
                '<div class="settings-nav-item" data-section="interface">Interface</div>' +
                '<div class="settings-nav-item" data-section="about">About</div>' +
            '</div>' +
            '<div class="settings-content" id="settings-content"></div>';

        panel.appendChild(layout);
    }

    function bindNav() {
        var nav = document.querySelector('.settings-nav');
        if (!nav) return;
        nav.addEventListener('click', function(e) {
            var item = e.target.closest('.settings-nav-item');
            if (!item) return;
            activeSection = item.dataset.section;
            nav.querySelectorAll('.settings-nav-item').forEach(function(n) {
                n.classList.toggle('active', n.dataset.section === activeSection);
            });
            renderSection(activeSection);
        });
    }

    function renderSection(section) {
        var content = document.getElementById('settings-content');
        if (!content) return;

        switch (section) {
            case 'general': content.innerHTML = buildGeneralSection(); bindGeneralEvents(); break;
            case 'memory': content.innerHTML = buildMemorySection(); bindMemoryEvents(); loadSystemInfo(); break;
            case 'output': content.innerHTML = buildOutputSection(); bindOutputEvents(); loadOutputFiles(); break;
            case 'interface': content.innerHTML = buildInterfaceSection(); bindInterfaceEvents(); break;
            case 'about': content.innerHTML = buildAboutSection(); loadAboutInfo(); break;
        }
    }

    // ── General Section ──
    function buildGeneralSection() {
        return '<div class="settings-section">' +
            '<h2 class="settings-section-title">General</h2>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Default Model</span>' +
                '<input type="text" id="settings-default-model" class="settings-input" value="' + esc(Settings.get('defaultModel')) + '" placeholder="Leave empty for first available">' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Default Steps</span>' +
                '<input type="number" id="settings-default-steps" class="settings-input-sm" min="1" max="150" value="' + Settings.get('defaultSteps') + '">' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Default CFG</span>' +
                '<input type="number" id="settings-default-cfg" class="settings-input-sm" min="1" max="20" step="0.5" value="' + Settings.get('defaultCfg') + '">' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Default Scheduler</span>' +
                '<select id="settings-default-scheduler" class="settings-select">' +
                    schedOpts() +
                '</select>' +
            '</div>' +
        '</div>';
    }

    function schedOpts() {
        var scheds = ['euler', 'euler_ancestral', 'dpm_2', 'dpmpp_2m', 'dpmpp_sde', 'ddim', 'lcm'];
        var current = Settings.get('defaultScheduler');
        return scheds.map(function(s) {
            return '<option value="' + s + '"' + (s === current ? ' selected' : '') + '>' + s + '</option>';
        }).join('');
    }

    function bindGeneralEvents() {
        bindInput('settings-default-model', 'defaultModel');
        bindInput('settings-default-steps', 'defaultSteps', 'int');
        bindInput('settings-default-cfg', 'defaultCfg', 'float');
        bindSelect('settings-default-scheduler', 'defaultScheduler');
    }

    // ── Memory Section ──
    function buildMemorySection() {
        var vram = Settings.get('vramBudgetMb');
        return '<div class="settings-section">' +
            '<h2 class="settings-section-title">Memory & Performance</h2>' +
            '<div class="settings-group">' +
                '<div class="settings-group-title">VRAM Budget</div>' +
                '<div class="settings-row">' +
                    '<input type="range" id="settings-vram-slider" class="settings-range" min="0" max="24576" value="' + vram + '">' +
                    '<span id="settings-vram-value" class="settings-range-value">' + (vram === 0 ? 'Auto' : formatMb(vram)) + '</span>' +
                '</div>' +
                '<div id="settings-gpu-name" class="settings-hint">Detecting GPU...</div>' +
            '</div>' +
            '<div class="settings-group">' +
                '<div class="settings-group-title">Pinned CPU Pool</div>' +
                '<div class="settings-row">' +
                    '<input type="range" id="settings-pinned-slider" class="settings-range" min="256" max="8192" step="256" value="' + Settings.get('pinned_pool_mb') + '">' +
                    '<span id="settings-pinned-value" class="settings-range-value">' + Settings.get('pinned_pool_mb') + ' MB</span>' +
                '</div>' +
                '<div class="settings-hint">Memory pinned for fast CPU\u2192GPU transfers.</div>' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">CPU Offload (Stagehand)</span>' +
                '<button id="settings-offload-toggle" class="settings-toggle' + (Settings.get('cpuOffloadEnabled') ? ' on' : '') + '">' +
                    (Settings.get('cpuOffloadEnabled') ? 'ON' : 'OFF') +
                '</button>' +
            '</div>' +
            '<hr class="settings-divider">' +
            '<div class="settings-group">' +
                '<div class="settings-group-title">Current Usage</div>' +
                '<div class="settings-bar-row">' +
                    '<span class="settings-bar-label">VRAM</span>' +
                    '<div class="settings-bar"><div id="settings-vram-bar" class="settings-bar-fill"></div></div>' +
                    '<span id="settings-vram-usage" class="settings-bar-value">-- / --</span>' +
                '</div>' +
                '<div class="settings-bar-row">' +
                    '<span class="settings-bar-label">RAM</span>' +
                    '<div class="settings-bar"><div id="settings-ram-bar" class="settings-bar-fill"></div></div>' +
                    '<span id="settings-ram-usage" class="settings-bar-value">-- / --</span>' +
                '</div>' +
                '<button id="settings-memory-refresh" class="settings-btn-sm">Refresh</button>' +
            '</div>' +
            '<button id="settings-memory-apply" class="settings-btn">Apply Memory Settings</button>' +
        '</div>';
    }

    function bindMemoryEvents() {
        var vramSlider = document.getElementById('settings-vram-slider');
        var vramValue = document.getElementById('settings-vram-value');
        if (vramSlider) {
            vramSlider.addEventListener('input', function() {
                var v = parseInt(this.value);
                vramValue.textContent = v === 0 ? 'Auto' : formatMb(v);
                Settings.set('vramBudgetMb', v);
            });
        }
        var pinnedSlider = document.getElementById('settings-pinned-slider');
        var pinnedValue = document.getElementById('settings-pinned-value');
        if (pinnedSlider) {
            pinnedSlider.addEventListener('input', function() {
                var v = parseInt(this.value);
                pinnedValue.textContent = v + ' MB';
                Settings.set('pinned_pool_mb', v);
            });
        }
        var offloadToggle = document.getElementById('settings-offload-toggle');
        if (offloadToggle) {
            offloadToggle.addEventListener('click', function() {
                var on = !Settings.get('cpuOffloadEnabled');
                Settings.set('cpuOffloadEnabled', on);
                this.classList.toggle('on', on);
                this.textContent = on ? 'ON' : 'OFF';
            });
        }
        var refreshBtn = document.getElementById('settings-memory-refresh');
        if (refreshBtn) refreshBtn.addEventListener('click', loadSystemInfo);
        var applyBtn = document.getElementById('settings-memory-apply');
        if (applyBtn) {
            applyBtn.addEventListener('click', function() {
                fetch('/stagehand_settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        vram_budget_mb: Settings.get('vramBudgetMb'),
                        pinned_pool_mb: Settings.get('pinned_pool_mb'),
                        cpu_offload: Settings.get('cpuOffloadEnabled')
                    })
                }).then(function() {
                    showSaved('Memory settings applied \u2014 takes effect on next model load.');
                }).catch(function() {
                    showSaved('Could not apply \u2014 server may not support runtime Stagehand config.');
                });
            });
        }
    }

    function loadSystemInfo() {
        fetch('/system_stats').then(function(r) { return r.json(); }).then(function(info) {
            var gpuLabel = document.getElementById('settings-gpu-name');
            if (gpuLabel) {
                if (info.devices && info.devices.length > 0) {
                    var dev = info.devices[0];
                    gpuLabel.textContent = 'Detected: ' + dev.name + ' (' + formatMb(dev.vram_total) + ')';
                    var slider = document.getElementById('settings-vram-slider');
                    if (slider) slider.max = dev.vram_total;
                    var vramUsed = dev.vram_total - (dev.vram_free || 0);
                    updateMemoryBars(vramUsed, dev.vram_total, 0, 0);
                } else {
                    gpuLabel.textContent = 'No GPU detected';
                }
            }
        }).catch(function() {
            var gpuLabel = document.getElementById('settings-gpu-name');
            if (gpuLabel) gpuLabel.textContent = 'Could not detect GPU';
        });
    }

    function updateMemoryBars(vramUsed, vramTotal, ramUsed, ramTotal) {
        var vramBar = document.getElementById('settings-vram-bar');
        var vramUsage = document.getElementById('settings-vram-usage');
        if (vramBar && vramTotal > 0) {
            vramBar.style.width = (vramUsed / vramTotal * 100) + '%';
        }
        if (vramUsage) {
            vramUsage.textContent = formatMb(vramUsed) + ' / ' + formatMb(vramTotal);
        }
        var ramBar = document.getElementById('settings-ram-bar');
        var ramUsage = document.getElementById('settings-ram-usage');
        if (ramBar && ramTotal > 0) {
            ramBar.style.width = (ramUsed / ramTotal * 100) + '%';
        }
        if (ramUsage && ramTotal > 0) {
            ramUsage.textContent = formatMb(ramUsed) + ' / ' + formatMb(ramTotal);
        }
    }

    // ── Output Section ──
    function buildOutputSection() {
        return '<div class="settings-section">' +
            '<h2 class="settings-section-title">Output & Files</h2>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Output Directory</span>' +
                '<input type="text" id="settings-output-dir" class="settings-input" value="' + esc(Settings.get('outputDir')) + '" placeholder="Server default">' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Image Format</span>' +
                '<div class="settings-pill-row">' +
                    formatPill('png') + formatPill('jpg') + formatPill('webp') +
                '</div>' +
            '</div>' +
            '<div id="settings-jpeg-row" class="settings-row" style="display:' + (Settings.get('outputFormat') === 'jpg' ? 'flex' : 'none') + '">' +
                '<span class="settings-label">JPEG Quality</span>' +
                '<input type="range" id="settings-jpeg-quality" class="settings-range" min="50" max="100" value="' + Settings.get('jpegQuality') + '">' +
                '<span id="settings-jpeg-val" class="settings-range-value">' + Settings.get('jpegQuality') + '</span>' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Save Metadata</span>' +
                '<button id="settings-metadata-toggle" class="settings-toggle' + (Settings.get('saveMetadata') ? ' on' : '') + '">' +
                    (Settings.get('saveMetadata') ? 'ON' : 'OFF') +
                '</button>' +
            '</div>' +
            '<hr class="settings-divider">' +
            '<div class="settings-group">' +
                '<div class="settings-row" style="justify-content:space-between">' +
                    '<span class="settings-group-title">Output Files</span>' +
                    '<button id="settings-open-folder" class="settings-btn-sm">Open Folder</button>' +
                '</div>' +
                '<div id="settings-file-list" class="settings-file-list">Loading...</div>' +
            '</div>' +
        '</div>';
    }

    function formatPill(fmt) {
        var active = Settings.get('outputFormat') === fmt ? ' active' : '';
        return '<button class="settings-pill' + active + '" data-format="' + fmt + '">' + fmt.toUpperCase() + '</button>';
    }

    function bindOutputEvents() {
        bindInput('settings-output-dir', 'outputDir');
        var jpegSlider = document.getElementById('settings-jpeg-quality');
        var jpegVal = document.getElementById('settings-jpeg-val');
        if (jpegSlider) {
            jpegSlider.addEventListener('input', function() {
                Settings.set('jpegQuality', parseInt(this.value));
                if (jpegVal) jpegVal.textContent = this.value;
            });
        }
        // Format pills
        var pillRow = document.querySelector('.settings-pill-row');
        if (pillRow) {
            pillRow.addEventListener('click', function(e) {
                var pill = e.target.closest('.settings-pill');
                if (!pill) return;
                var fmt = pill.dataset.format;
                Settings.set('outputFormat', fmt);
                pillRow.querySelectorAll('.settings-pill').forEach(function(p) {
                    p.classList.toggle('active', p.dataset.format === fmt);
                });
                var jpegRow = document.getElementById('settings-jpeg-row');
                if (jpegRow) jpegRow.style.display = fmt === 'jpg' ? 'flex' : 'none';
            });
        }
        var metaToggle = document.getElementById('settings-metadata-toggle');
        if (metaToggle) {
            metaToggle.addEventListener('click', function() {
                var on = !Settings.get('saveMetadata');
                Settings.set('saveMetadata', on);
                this.classList.toggle('on', on);
                this.textContent = on ? 'ON' : 'OFF';
            });
        }
        var openFolder = document.getElementById('settings-open-folder');
        if (openFolder) {
            openFolder.addEventListener('click', function() {
                fetch('/open_output_dir', { method: 'POST' }).catch(function() {});
            });
        }
    }

    function loadOutputFiles() {
        fetch('/output_files').then(function(r) { return r.json(); }).then(function(files) {
            var list = document.getElementById('settings-file-list');
            if (!list) return;
            if (!files || !files.length) { list.innerHTML = '<div class="settings-empty">No output files</div>'; return; }
            list.innerHTML = '';
            files.forEach(function(f) {
                var row = document.createElement('div');
                row.className = 'settings-file-row';
                row.innerHTML =
                    '<span class="settings-file-name">' + esc(f.name) + '</span>' +
                    '<span class="settings-file-size">' + formatBytes(f.size_bytes) + '</span>' +
                    '<span class="settings-file-time">' + formatTimeAgo(f.modified * 1000) + '</span>' +
                    '<a class="settings-file-btn" href="/view?filename=' + encodeURIComponent(f.name) + '&type=output" download title="Download">\u2193</a>' +
                    '<button class="settings-file-btn settings-file-delete" data-name="' + esc(f.name) + '" title="Delete">\u2717</button>';
                list.appendChild(row);
            });
            list.onclick = function(e) {
                var del = e.target.closest('.settings-file-delete');
                if (!del) return;
                var name = del.dataset.name;
                fetch('/output_files/' + encodeURIComponent(name), { method: 'DELETE' }).then(function(r) {
                    if (r.ok) del.closest('.settings-file-row').remove();
                });
            };
        }).catch(function() {
            var list = document.getElementById('settings-file-list');
            if (list) list.innerHTML = '<div class="settings-empty">Could not load files</div>';
        });
    }

    // ── Interface Section ──
    function buildInterfaceSection() {
        var s = Settings.getAll();
        var swatches = ACCENT_PRESETS.map(function(p) {
            var active = s.accentColor === p.hex ? ' active' : '';
            return '<button class="settings-swatch' + active + '" data-color="' + p.hex + '" style="background:' + p.hex + '" title="' + p.label + '"></button>';
        }).join('');

        return '<div class="settings-section">' +
            '<h2 class="settings-section-title">Interface</h2>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Theme</span>' +
                '<div class="settings-pill-row">' +
                    themePill('dark') + themePill('light') + themePill('system') +
                '</div>' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Accent Color</span>' +
                '<div class="settings-swatch-row">' + swatches +
                    '<input type="text" id="settings-custom-accent" class="settings-input-sm" value="' + s.accentColor + '" maxlength="7" style="width:70px">' +
                '</div>' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Default Mode</span>' +
                '<div class="settings-pill-row">' +
                    modePill('simple', 'Simple') + modePill('advanced', 'Advanced') +
                '</div>' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Default Tab</span>' +
                '<select id="settings-default-tab" class="settings-select">' +
                    tabOpts() +
                '</select>' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Gallery Columns</span>' +
                '<div class="settings-pill-row">' +
                    colPill(2) + colPill(3) + colPill(4) +
                '</div>' +
            '</div>' +
            '<div class="settings-row">' +
                '<span class="settings-label">Confirm on Clear</span>' +
                '<button id="settings-confirm-toggle" class="settings-toggle' + (s.confirmOnClear ? ' on' : '') + '">' +
                    (s.confirmOnClear ? 'ON' : 'OFF') +
                '</button>' +
            '</div>' +
            '<hr class="settings-divider">' +
            '<button id="settings-reset-btn" class="settings-btn settings-btn-danger">Reset All Settings to Default</button>' +
        '</div>';
    }

    function themePill(t) {
        var active = Settings.get('theme') === t ? ' active' : '';
        return '<button class="settings-pill' + active + '" data-theme="' + t + '">' + t.charAt(0).toUpperCase() + t.slice(1) + '</button>';
    }

    function modePill(m, label) {
        var active = Settings.get('defaultMode') === m ? ' active' : '';
        return '<button class="settings-pill' + active + '" data-mode="' + m + '">' + label + '</button>';
    }

    function colPill(n) {
        var active = Settings.get('galleryColumns') === n ? ' active' : '';
        return '<button class="settings-pill' + active + '" data-cols="' + n + '">' + n + '</button>';
    }

    function tabOpts() {
        var tabs = ['generate', 'canvas', 'workflows', 'queue', 'models'];
        var current = Settings.get('defaultTab');
        return tabs.map(function(t) {
            return '<option value="' + t + '"' + (t === current ? ' selected' : '') + '>' + t.charAt(0).toUpperCase() + t.slice(1) + '</option>';
        }).join('');
    }

    function bindInterfaceEvents() {
        // Theme pills
        document.querySelectorAll('[data-theme]').forEach(function(pill) {
            pill.addEventListener('click', function() {
                Settings.set('theme', this.dataset.theme);
                document.querySelectorAll('[data-theme]').forEach(function(p) {
                    p.classList.toggle('active', p.dataset.theme === Settings.get('theme'));
                });
            });
        });
        // Accent swatches
        document.querySelectorAll('.settings-swatch').forEach(function(sw) {
            sw.addEventListener('click', function() {
                Settings.set('accentColor', this.dataset.color);
                document.querySelectorAll('.settings-swatch').forEach(function(s) {
                    s.classList.toggle('active', s.dataset.color === Settings.get('accentColor'));
                });
                var customInput = document.getElementById('settings-custom-accent');
                if (customInput) customInput.value = Settings.get('accentColor');
            });
        });
        // Custom accent
        var customAccent = document.getElementById('settings-custom-accent');
        if (customAccent) {
            customAccent.addEventListener('blur', function() {
                if (/^#[0-9a-fA-F]{6}$/.test(this.value)) {
                    Settings.set('accentColor', this.value);
                    document.querySelectorAll('.settings-swatch').forEach(function(s) {
                        s.classList.toggle('active', s.dataset.color === Settings.get('accentColor'));
                    });
                }
            });
        }
        // Mode pills
        document.querySelectorAll('[data-mode]').forEach(function(pill) {
            pill.addEventListener('click', function() {
                Settings.set('defaultMode', this.dataset.mode);
                document.querySelectorAll('[data-mode]').forEach(function(p) {
                    p.classList.toggle('active', p.dataset.mode === Settings.get('defaultMode'));
                });
            });
        });
        // Column pills
        document.querySelectorAll('[data-cols]').forEach(function(pill) {
            pill.addEventListener('click', function() {
                Settings.set('galleryColumns', parseInt(this.dataset.cols));
                document.querySelectorAll('[data-cols]').forEach(function(p) {
                    p.classList.toggle('active', parseInt(p.dataset.cols) === Settings.get('galleryColumns'));
                });
            });
        });
        bindSelect('settings-default-tab', 'defaultTab');
        var confirmToggle = document.getElementById('settings-confirm-toggle');
        if (confirmToggle) {
            confirmToggle.addEventListener('click', function() {
                var on = !Settings.get('confirmOnClear');
                Settings.set('confirmOnClear', on);
                this.classList.toggle('on', on);
                this.textContent = on ? 'ON' : 'OFF';
            });
        }
        var resetBtn = document.getElementById('settings-reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', function() {
                if (confirm('Reset all settings to defaults? This will reload the page.')) {
                    Settings.reset();
                }
            });
        }
    }

    // ── About Section ──
    function buildAboutSection() {
        return '<div class="settings-section">' +
            '<h2 class="settings-section-title">About SerenityFlow</h2>' +
            '<div class="settings-about-header">' +
                '<div class="settings-about-icon">\u2726</div>' +
                '<div><div class="settings-about-name">SerenityFlow</div>' +
                '<div class="settings-about-tagline">Stagehand-native inference engine</div></div>' +
            '</div>' +
            '<div class="settings-about-version">Version: v2.0.0-beta</div>' +
            '<hr class="settings-divider">' +
            '<div class="settings-group">' +
                '<div class="settings-group-title">Backend</div>' +
                '<div id="settings-about-info" class="settings-about-info">Loading...</div>' +
            '</div>' +
            '<hr class="settings-divider">' +
            '<button id="settings-copy-debug" class="settings-btn-sm">Copy Debug Info</button>' +
        '</div>';
    }

    function loadAboutInfo() {
        fetch('/system_stats').then(function(r) { return r.json(); }).then(function(info) {
            var el = document.getElementById('settings-about-info');
            if (!el) return;
            var gpu = info.devices && info.devices.length > 0 ? info.devices[0] : null;
            var sys = info.system || {};
            el.innerHTML =
                '<div>GPU: ' + (gpu ? gpu.name + ' (' + formatMb(gpu.vram_total) + ')' : 'None') + '</div>' +
                '<div>Python: ' + (sys.python_version || 'N/A') + '</div>' +
                '<div>PyTorch: ' + (sys.pytorch_version || 'N/A') + '</div>' +
                '<div>CUDA: ' + (sys.cuda_version || 'N/A') + '</div>';

            var copyBtn = document.getElementById('settings-copy-debug');
            if (copyBtn) {
                copyBtn.onclick = function() {
                    var text = 'SerenityFlow v2.0.0-beta\n' +
                        'GPU: ' + (gpu ? gpu.name + ' (' + formatMb(gpu.vram_total) + ')' : 'None') + '\n' +
                        'Python: ' + (sys.python_version || 'N/A') + '\n' +
                        'PyTorch: ' + (sys.pytorch_version || 'N/A') + '\n' +
                        'CUDA: ' + (sys.cuda_version || 'N/A');
                    navigator.clipboard.writeText(text).then(function() {
                        copyBtn.textContent = 'Copied!';
                        setTimeout(function() { copyBtn.textContent = 'Copy Debug Info'; }, 1500);
                    });
                };
            }
        }).catch(function() {
            var el = document.getElementById('settings-about-info');
            if (el) el.textContent = 'Could not load system info';
        });
    }

    // ── Helpers ──
    function bindInput(id, key, type) {
        var el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('blur', function() {
            var v = this.value;
            if (type === 'int') v = parseInt(v) || Settings.DEFAULTS[key];
            else if (type === 'float') v = parseFloat(v) || Settings.DEFAULTS[key];
            Settings.set(key, v);
        });
    }

    function bindSelect(id, key) {
        var el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('change', function() { Settings.set(key, this.value); });
    }

    function esc(str) {
        if (!str) return '';
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function formatMb(mb) { return (mb / 1024).toFixed(1) + ' GB'; }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function formatTimeAgo(ts) {
        var diff = Math.floor((Date.now() - ts) / 1000);
        if (diff < 60) return 'just now';
        if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
        return Math.floor(diff / 86400) + 'd ago';
    }

    function showSaved(msg) {
        var el = document.createElement('div');
        el.className = 'settings-toast';
        el.textContent = msg;
        document.body.appendChild(el);
        setTimeout(function() { el.classList.add('visible'); }, 10);
        setTimeout(function() { el.classList.remove('visible'); setTimeout(function() { el.remove(); }, 300); }, 3000);
    }

    function init() {
        if (initialized) return;
        initialized = true;
        buildUI();
        bindNav();
        renderSection('general');
    }

    return { init: init };
})();
