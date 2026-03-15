/**
 * Queue Tab — SerenityFlow Phase 6
 * Real-time job tracking with pending queue, current job, and history.
 */
var QueueTab = (function() {
    'use strict';

    var initialized = false;
    var MAX_HISTORY = 100;

    var state = {
        current: null,
        pending: [],
        history: []
    };

    var els = {};
    var elapsedTimer = null;

    function buildUI() {
        var panel = document.getElementById('panel-queue');
        if (!panel) return;
        panel.innerHTML = '';

        var layout = document.createElement('div');
        layout.className = 'queue-layout';
        layout.innerHTML =
            '<div class="queue-header">' +
                '<span class="queue-header-title">Queue</span>' +
                '<button id="queue-clear-btn" class="queue-clear-btn">Clear Finished</button>' +
            '</div>' +
            '<div class="queue-sections">' +
                '<div class="queue-section">' +
                    '<div class="queue-section-label">Current</div>' +
                    '<div id="queue-current-card" class="queue-current-card">' +
                        '<div class="queue-empty-state">No job running</div>' +
                    '</div>' +
                '</div>' +
                '<div class="queue-section" id="queue-pending-section" style="display:none">' +
                    '<div class="queue-section-label">Pending <span id="queue-pending-count"></span></div>' +
                    '<div id="queue-pending-list"></div>' +
                '</div>' +
                '<div class="queue-section">' +
                    '<div class="queue-section-label">History</div>' +
                    '<div id="queue-history-list"></div>' +
                '</div>' +
            '</div>';

        panel.appendChild(layout);

        els.currentCard = document.getElementById('queue-current-card');
        els.pendingSection = document.getElementById('queue-pending-section');
        els.pendingCount = document.getElementById('queue-pending-count');
        els.pendingList = document.getElementById('queue-pending-list');
        els.historyList = document.getElementById('queue-history-list');
        els.clearBtn = document.getElementById('queue-clear-btn');
    }

    function bindEvents() {
        els.clearBtn.addEventListener('click', function() {
            state.history = [];
            renderHistory();
            saveHistory();
        });
    }

    function connectWS() {
        SerenityWS.on('execution_start', function(data) {
            var id = data && data.prompt_id ? data.prompt_id : null;
            // Find matching pending entry
            var found = null;
            state.pending = state.pending.filter(function(p) {
                if (p.promptId === id) { found = p; return false; }
                return true;
            });
            state.current = {
                promptId: id,
                prompt: found ? found.prompt : '',
                model: found ? found.model : '',
                batchLabel: found ? found.batchLabel : '',
                startedAt: Date.now(),
                step: 0, maxStep: 0, node: null, src: null, filename: null
            };
            startElapsedTimer();
            render();
        });

        SerenityWS.on('executing', function(data) {
            if (!state.current || !data) return;
            if (data.node) state.current.node = data.node;
            renderCurrentJob();
        });

        SerenityWS.on('progress', function(data) {
            if (!state.current || !data) return;
            state.current.step = data.value;
            state.current.maxStep = data.max;
            renderCurrentJob();
        });

        SerenityWS.on('executed', function(data) {
            if (!state.current || !data || !data.output) return;
            var out = data.output.ui || data.output;
            var items = out.images || out.videos;
            if (items && items.length > 0) {
                var file = items[0];
                state.current.src = SerenityAPI.viewUrl(file.filename, file.subfolder, file.type);
                state.current.filename = file.filename;
            }
        });

        SerenityWS.on('execution_success', function() {
            if (!state.current) return;
            stopElapsedTimer();
            state.history.unshift({
                promptId: state.current.promptId,
                prompt: state.current.prompt,
                model: state.current.model,
                batchLabel: state.current.batchLabel,
                status: 'success',
                completedAt: Date.now(),
                filename: state.current.filename,
                src: state.current.src
            });
            if (state.history.length > MAX_HISTORY) state.history.pop();
            state.current = null;
            saveHistory();
            render();
        });

        SerenityWS.on('execution_error', function(data) {
            if (!state.current) return;
            stopElapsedTimer();
            state.history.unshift({
                promptId: state.current.promptId,
                prompt: state.current.prompt,
                model: state.current.model,
                status: 'error',
                error: (data && data.exception_message) || 'Unknown error',
                completedAt: Date.now()
            });
            if (state.history.length > MAX_HISTORY) state.history.pop();
            state.current = null;
            saveHistory();
            render();
        });
    }

    function startElapsedTimer() {
        stopElapsedTimer();
        elapsedTimer = setInterval(function() {
            if (state.current) renderCurrentJob();
        }, 1000);
    }

    function stopElapsedTimer() {
        if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    }

    // ── Rendering ──

    function render() {
        renderCurrentJob();
        renderPending();
        renderHistory();
        updateQueueBadge();
    }

    function renderCurrentJob() {
        if (!els.currentCard) return;
        if (!state.current) {
            els.currentCard.innerHTML = '<div class="queue-empty-state">No job running</div>';
            return;
        }
        var c = state.current;
        var pct = c.maxStep > 0 ? Math.round(c.step / c.maxStep * 100) : 0;
        var elapsed = Math.round((Date.now() - c.startedAt) / 1000);
        var label = c.batchLabel ? c.batchLabel + ' · ' : '';

        els.currentCard.innerHTML =
            '<div class="queue-job-card current">' +
                (c.src ? '<img class="queue-thumb" src="' + c.src + '">' : '<div class="queue-thumb-placeholder"></div>') +
                '<div class="queue-job-info">' +
                    '<div class="queue-job-model">' + label + escapeHtml(c.model || 'Unknown') + '</div>' +
                    '<div class="queue-job-prompt">' + escapeHtml(truncate(c.prompt, 60)) + '</div>' +
                    '<div class="queue-progress-wrap">' +
                        '<div class="queue-progress-bar"><div class="queue-progress-fill" style="width:' + pct + '%"></div></div>' +
                    '</div>' +
                    '<div class="queue-progress-text">Step ' + c.step + ' / ' + c.maxStep + ' · ' + pct + '% · ' + elapsed + 's' + (c.node ? ' · ' + c.node : '') + '</div>' +
                '</div>' +
                '<button class="queue-cancel-btn" id="queue-cancel-current">Cancel</button>' +
            '</div>';

        var cancelBtn = document.getElementById('queue-cancel-current');
        if (cancelBtn) {
            cancelBtn.onclick = function() { SerenityAPI.interrupt(); };
        }
    }

    function renderPending() {
        if (!els.pendingSection || !els.pendingList) return;
        if (state.pending.length === 0) {
            els.pendingSection.style.display = 'none';
            return;
        }
        els.pendingSection.style.display = 'block';
        els.pendingCount.textContent = '(' + state.pending.length + ')';
        els.pendingList.innerHTML = '';
        state.pending.forEach(function(p, idx) {
            var row = document.createElement('div');
            row.className = 'queue-pending-row';
            row.innerHTML =
                '<span class="queue-pending-num">#' + (idx + 1) + '</span>' +
                '<span class="queue-pending-prompt">' + escapeHtml(truncate(p.prompt, 50)) + '</span>' +
                (p.batchLabel ? '<span class="queue-pending-batch">' + p.batchLabel + '</span>' : '') +
                '<button class="queue-pending-remove" data-id="' + p.promptId + '">Remove</button>';
            els.pendingList.appendChild(row);
        });
        els.pendingList.onclick = function(e) {
            var btn = e.target.closest('.queue-pending-remove');
            if (!btn) return;
            var id = btn.dataset.id;
            state.pending = state.pending.filter(function(p) { return p.promptId !== id; });
            renderPending();
            updateQueueBadge();
        };
    }

    function renderHistory() {
        if (!els.historyList) return;
        if (state.history.length === 0) {
            els.historyList.innerHTML = '<div class="queue-empty-state">No history</div>';
            return;
        }
        els.historyList.innerHTML = '';
        state.history.forEach(function(entry) {
            var row = document.createElement('div');
            row.className = 'queue-history-entry ' + entry.status;
            var isSuccess = entry.status === 'success';
            var timeAgo = formatTimeAgo(entry.completedAt);

            row.innerHTML =
                '<span class="queue-status-icon">' + (isSuccess ? '\u2713' : '\u2717') + '</span>' +
                (isSuccess && entry.src ? '<img class="queue-thumb-sm" src="' + entry.src + '">' : '') +
                '<div class="queue-history-info">' +
                    '<div class="queue-history-filename">' + escapeHtml(entry.filename || entry.model || 'Unknown') + '</div>' +
                    (entry.batchLabel ? '<span class="queue-history-batch">' + entry.batchLabel + '</span>' : '') +
                    (!isSuccess && entry.error ? '<div class="queue-history-error">' + escapeHtml(entry.error) + '</div>' : '') +
                '</div>' +
                '<span class="queue-history-time">' + timeAgo + '</span>' +
                '<div class="queue-history-actions">' +
                    (isSuccess && entry.src ? '<button class="queue-action-btn queue-view-btn" data-src="' + entry.src + '" data-video="' + (/\.(webp|mp4|gif)$/i.test(entry.filename || '') ? '1' : '0') + '">View</button>' : '') +
                    '<button class="queue-action-btn queue-dismiss-btn" data-id="' + entry.promptId + '">\u00d7</button>' +
                '</div>';

            els.historyList.appendChild(row);
        });

        els.historyList.onclick = function(e) {
            var viewBtn = e.target.closest('.queue-view-btn');
            if (viewBtn) {
                localStorage.setItem('sf-view-image', JSON.stringify({
                    src: viewBtn.dataset.src,
                    isVideo: viewBtn.dataset.video === '1'
                }));
                if (typeof switchTab === 'function') switchTab('generate');
                return;
            }
            var dismissBtn = e.target.closest('.queue-dismiss-btn');
            if (dismissBtn) {
                var id = dismissBtn.dataset.id;
                state.history = state.history.filter(function(h) { return h.promptId !== id; });
                renderHistory();
                saveHistory();
            }
        };
    }

    // ── Queue Badge on Icon Rail ──

    function updateQueueBadge() {
        var badge = document.getElementById('queue-rail-badge');
        if (!badge) return;
        var running = state.current !== null;
        var pending = state.pending.length;
        if (running || pending > 0) {
            badge.style.display = 'flex';
            badge.textContent = running ? (pending + 1) : pending;
        } else {
            badge.style.display = 'none';
        }
    }

    // ── Helpers ──

    function truncate(str, max) {
        if (!str) return '';
        return str.length > max ? str.substring(0, max) + '...' : str;
    }

    function escapeHtml(str) {
        if (!str) return '';
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function formatTimeAgo(timestamp) {
        if (!timestamp) return '';
        var diff = Math.floor((Date.now() - timestamp) / 1000);
        if (diff < 60) return 'just now';
        if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
        return Math.floor(diff / 86400) + 'd ago';
    }

    function saveHistory() {
        try {
            localStorage.setItem('sf-queue-history', JSON.stringify(state.history.slice(0, MAX_HISTORY)));
        } catch(e) {}
    }

    function loadHistory() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-queue-history'));
            if (saved && Array.isArray(saved)) state.history = saved.slice(0, MAX_HISTORY);
        } catch(e) { state.history = []; }
    }

    // ── Public ──

    function registerPending(entry) {
        state.pending.push(entry);
        render();
    }

    function init() {
        if (initialized) return;
        initialized = true;
        buildUI();
        bindEvents();
        loadHistory();
        connectWS();
        render();
    }

    return {
        init: init,
        registerPending: registerPending,
        state: state
    };
})();
