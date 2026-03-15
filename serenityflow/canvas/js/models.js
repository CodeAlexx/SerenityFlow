/**
 * Models Tab — SerenityFlow Phase 6
 * Model browser with search, filter, and architecture detection.
 */
var ModelsTab = (function() {
    'use strict';

    var initialized = false;
    var allModels = [];
    var filteredModels = [];
    var filters = { search: '', type: 'all', archs: {} };

    var ARCH_COLORS = {
        flux: '#6c6af5',
        sdxl: '#3b82f6',
        sd3: '#8b5cf6',
        sd15: '#6b7280',
        ltxv: '#10b981',
        wan: '#f59e0b',
        klein: '#ec4899',
        any: '#6b7280'
    };

    function estimateSize(filename) {
        var f = filename.toLowerCase();
        if (f.includes('flux') && f.includes('fp8')) return '~11 GB';
        if (f.includes('flux')) return '~22 GB';
        if (f.includes('sdxl') || f.includes('xl') || f.includes('pony') || f.includes('illustrious')) return '~7 GB';
        if (f.includes('sd3')) return '~5 GB';
        if (f.includes('ltx')) return '~19 GB';
        if (f.includes('wan')) return '~14 GB';
        if (f.includes('lora')) return '~150 MB';
        if (f.includes('controlnet') || f.includes('control')) return '~1.5 GB';
        if (f.includes('vae') || f.includes('ae.')) return '~350 MB';
        return '~2 GB';
    }

    function buildUI() {
        var panel = document.getElementById('panel-models');
        if (!panel) return;
        panel.innerHTML = '';

        var layout = document.createElement('div');
        layout.className = 'models-layout';
        layout.innerHTML =
            '<div class="models-header">' +
                '<span class="models-header-title">Models</span>' +
                '<div class="models-header-actions">' +
                    '<input type="text" id="models-search" class="models-search" placeholder="Search models...">' +
                    '<button id="models-refresh-btn" class="models-refresh-btn" title="Refresh">&#8635;</button>' +
                '</div>' +
            '</div>' +
            '<div class="models-body">' +
                '<div class="models-sidebar">' +
                    '<div class="models-filter-group">' +
                        '<div class="models-filter-title">Type</div>' +
                        '<label class="models-radio"><input type="radio" name="model-type" value="all" checked> All</label>' +
                        '<label class="models-radio"><input type="radio" name="model-type" value="checkpoint"> Checkpoints</label>' +
                        '<label class="models-radio"><input type="radio" name="model-type" value="unet"> UNets</label>' +
                        '<label class="models-radio"><input type="radio" name="model-type" value="lora"> LoRA</label>' +
                        '<label class="models-radio"><input type="radio" name="model-type" value="vae"> VAE</label>' +
                        '<label class="models-radio"><input type="radio" name="model-type" value="controlnet"> ControlNet</label>' +
                    '</div>' +
                    '<div class="models-filter-group">' +
                        '<div class="models-filter-title">Architecture</div>' +
                        '<div id="models-arch-filters"></div>' +
                    '</div>' +
                    '<div class="models-count" id="models-count">0 models</div>' +
                '</div>' +
                '<div class="models-grid" id="models-grid"></div>' +
            '</div>';

        panel.appendChild(layout);
    }

    function bindEvents() {
        var search = document.getElementById('models-search');
        if (search) {
            search.addEventListener('input', function() {
                filters.search = this.value.toLowerCase();
                applyFilters();
            });
        }

        var refreshBtn = document.getElementById('models-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', function() {
                ModelUtils.clearCache();
                load();
            });
        }

        // Type filter radios
        document.querySelectorAll('input[name="model-type"]').forEach(function(radio) {
            radio.addEventListener('change', function() {
                filters.type = this.value;
                applyFilters();
            });
        });
    }

    function load() {
        allModels = [];
        return fetch('/object_info', { cache: 'no-store' })
            .then(function(resp) { return resp.ok ? resp.json() : {}; })
            .then(function(info) {
                var seen = {};
                function addModels(nodeType, inputKey, type, defaultArch) {
                    var items = info && info[nodeType] && info[nodeType].input && info[nodeType].input.required && info[nodeType].input.required[inputKey];
                    if (items && Array.isArray(items[0])) {
                        items[0].forEach(function(name) {
                            if (seen[name]) return;
                            seen[name] = true;
                            allModels.push({
                                name: name,
                                type: type,
                                arch: defaultArch || ModelUtils.detectArchFromFilename(name)
                            });
                        });
                    }
                }

                addModels('CheckpointLoaderSimple', 'ckpt_name', 'checkpoint');
                addModels('UNETLoader', 'unet_name', 'unet');
                addModels('LoraLoader', 'lora_name', 'lora');
                addModels('VAELoader', 'vae_name', 'vae', 'any');
                addModels('ControlNetLoader', 'control_net_name', 'controlnet');

                buildArchFilters();
                applyFilters();
            })
            .catch(function(err) {
                console.error('ModelsTab load failed:', err);
                renderGrid([]);
            });
    }

    function buildArchFilters() {
        var archSet = {};
        allModels.forEach(function(m) { archSet[m.arch] = true; });
        var container = document.getElementById('models-arch-filters');
        if (!container) return;
        container.innerHTML = '';
        Object.keys(archSet).sort().forEach(function(arch) {
            var label = document.createElement('label');
            label.className = 'models-checkbox';
            label.innerHTML = '<input type="checkbox" data-arch="' + arch + '" checked> ' + arch.toUpperCase();
            container.appendChild(label);
            filters.archs[arch] = true;
        });
        container.addEventListener('change', function(e) {
            var cb = e.target;
            if (cb.dataset.arch) {
                filters.archs[cb.dataset.arch] = cb.checked;
                applyFilters();
            }
        });
    }

    function applyFilters() {
        filteredModels = allModels.filter(function(m) {
            if (filters.search && m.name.toLowerCase().indexOf(filters.search) === -1) return false;
            if (filters.type !== 'all' && m.type !== filters.type) return false;
            if (!filters.archs[m.arch]) return false;
            return true;
        });
        renderGrid(filteredModels);
        var countEl = document.getElementById('models-count');
        if (countEl) countEl.textContent = filteredModels.length + ' model' + (filteredModels.length !== 1 ? 's' : '');
    }

    function renderGrid(models) {
        var grid = document.getElementById('models-grid');
        if (!grid) return;
        if (models.length === 0) {
            grid.innerHTML = '<div class="models-empty">No models found</div>';
            return;
        }
        grid.innerHTML = '';
        models.forEach(function(m) {
            var card = document.createElement('div');
            card.className = 'model-card';
            var color = ARCH_COLORS[m.arch] || ARCH_COLORS.any;
            card.innerHTML =
                '<div class="model-card-badge" style="background:' + color + '">' + m.arch.toUpperCase() + '</div>' +
                '<div class="model-card-type">' + m.type + '</div>' +
                '<div class="model-card-name" title="' + escapeHtml(m.name) + '">' + escapeHtml(m.name) + '</div>' +
                '<div class="model-card-size">' + estimateSize(m.name) + '</div>' +
                '<button class="model-use-btn" data-name="' + escapeHtml(m.name) + '">Use in Generate</button>';
            grid.appendChild(card);
        });
        grid.onclick = function(e) {
            var btn = e.target.closest('.model-use-btn');
            if (!btn) return;
            useModelInGenerate(btn.dataset.name);
        };
    }

    function useModelInGenerate(modelName) {
        if (typeof GenerateTab !== 'undefined' && GenerateTab.state) {
            GenerateTab.state.model = modelName;
            var picker = document.getElementById('gen-model');
            if (picker) picker.value = modelName;
        }
        if (typeof switchTab === 'function') switchTab('generate');
    }

    function escapeHtml(str) {
        if (!str) return '';
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function init() {
        if (initialized) return;
        initialized = true;
        buildUI();
        bindEvents();
        load();
    }

    return { init: init, load: load };
})();
