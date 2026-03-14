/**
 * Queue Tab — SerenityFlow
 * Real-time job status, progress tracking, and history.
 */

var QueueTab = (function() {
    'use strict';

    var initialized = false;
    var history = [];
    var currentJob = null;
    var els = {};

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
            '<div id="queue-current" class="queue-current">' +
                '<div class="queue-current-thumb" id="queue-current-thumb">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="color:var(--shell-text-muted)"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>' +
                '</div>' +
                '<div class="queue-current-info">' +
                    '<div class="queue-current-status">' +
                        '<span class="queue-current-dot"></span>' +
                        '<span id="queue-current-label">Running</span>' +
                    '</div>' +
                    '<div id="queue-current-node" class="queue-current-node"></div>' +
                    '<div class="queue-progress-row">' +
                        '<span id="queue-progress-text" class="queue-progress-text">Step 0 / 0</span>' +
                        '<div class="queue-progress-track"><div id="queue-progress-fill" class="queue-progress-fill"></div></div>' +
                        '<span id="queue-progress-pct" class="queue-progress-pct">0%</span>' +
                    '</div>' +
                    '<button id="queue-cancel" class="queue-cancel-btn">Cancel</button>' +
                '</div>' +
            '</div>' +
            '<div class="queue-history-label">History</div>' +
            '<div id="queue-history" class="queue-history">' +
                '<div class="queue-history-empty">No recent jobs</div>' +
            '</div>';

        panel.appendChild(layout);
        cacheElements();
    }

    function cacheElements() {
        els.currentSection = document.getElementById('queue-current');
        els.currentThumb = document.getElementById('queue-current-thumb');
        els.currentLabel = document.getElementById('queue-current-label');
        els.currentNode = document.getElementById('queue-current-node');
        els.progressText = document.getElementById('queue-progress-text');
        els.progressFill = document.getElementById('queue-progress-fill');
        els.progressPct = document.getElementById('queue-progress-pct');
        els.cancelBtn = document.getElementById('queue-cancel');
        els.historyList = document.getElementById('queue-history');
        els.clearBtn = document.getElementById('queue-clear-btn');
    }

    function bindEvents() {
        els.cancelBtn.addEventListener('click', function() {
            fetch('/interrupt', { method: 'POST' })
                .catch(function(err) {
                    console.error('Failed to cancel:', err);
                });
        });

        els.clearBtn.addEventListener('click', function() {
            history = [];
            renderHistory();
            saveHistory();
        });
    }

    function connectWS() {
        SerenityWS.on('execution_start', function(data) {
            currentJob = {
                id: data && data.prompt_id ? data.prompt_id : Date.now(),
                startTime: Date.now(),
                step: 0,
                maxSteps: 0,
                node: ''
            };
            els.currentSection.classList.add('active');
            els.currentLabel.textContent = 'Running';
            els.currentNode.textContent = '';
            els.progressText.textContent = 'Starting...';
            els.progressFill.style.width = '0%';
            els.progressPct.textContent = '0%';
        });

        SerenityWS.on('executing', function(data) {
            if (!data || !currentJob) return;
            if (data.node) {
                els.currentNode.textContent = 'Node: ' + data.node;
            }
        });

        SerenityWS.on('progress', function(data) {
            if (!data || !currentJob) return;
            currentJob.step = data.value;
            currentJob.maxSteps = data.max;
            var pct = Math.round(data.value / data.max * 100);
            els.progressText.textContent = 'Step ' + data.value + ' / ' + data.max;
            els.progressFill.style.width = pct + '%';
            els.progressPct.textContent = pct + '%';
        });

        SerenityWS.on('executed', function(data) {
            if (!currentJob) return;
            // Extract filename for history
            var filename = 'Unknown';
            var thumbSrc = null;
            if (data && data.output) {
                var items = data.output.images || data.output.videos;
                if (items && items.length > 0) {
                    filename = items[0].filename || 'output';
                    thumbSrc = '/view?filename=' + encodeURIComponent(items[0].filename) +
                        '&subfolder=' + encodeURIComponent(items[0].subfolder || '') +
                        '&type=' + encodeURIComponent(items[0].type || 'output');
                }
            }
            // Don't hide current yet — wait for execution_success
            currentJob.filename = filename;
            currentJob.thumbSrc = thumbSrc;
        });

        SerenityWS.on('execution_success', function() {
            if (!currentJob) return;
            addToHistory({
                filename: currentJob.filename || 'output',
                thumbSrc: currentJob.thumbSrc,
                status: 'success',
                time: Date.now()
            });
            currentJob = null;
            els.currentSection.classList.remove('active');
        });

        SerenityWS.on('execution_error', function(data) {
            var errMsg = (data && data.exception_message) || 'Unknown error';
            addToHistory({
                filename: currentJob ? (currentJob.filename || 'failed job') : 'failed job',
                status: 'error',
                error: errMsg,
                time: Date.now()
            });
            currentJob = null;
            els.currentSection.classList.remove('active');
        });
    }

    function addToHistory(item) {
        history.unshift(item);
        if (history.length > 50) history.pop();
        renderHistory();
        saveHistory();
    }

    function renderHistory() {
        if (!els.historyList) return;

        if (history.length === 0) {
            els.historyList.innerHTML = '<div class="queue-history-empty">No recent jobs</div>';
            return;
        }

        els.historyList.innerHTML = '';
        history.forEach(function(item, idx) {
            var row = document.createElement('div');
            row.className = 'queue-item';

            var isSuccess = item.status === 'success';
            var icon = isSuccess ? '\u2713' : '\u2717';
            var iconClass = isSuccess ? 'success' : 'error';
            var timeAgo = formatTimeAgo(item.time);

            var html =
                '<span class="queue-item-icon ' + iconClass + '">' + icon + '</span>' +
                '<div style="flex:1;min-width:0">' +
                    '<div class="queue-item-name">' + escapeHtml(item.filename) + '</div>' +
                    (item.error ? '<div class="queue-item-error">' + escapeHtml(item.error) + '</div>' : '') +
                '</div>' +
                '<span class="queue-item-time">' + timeAgo + '</span>' +
                '<div class="queue-item-actions">';

            if (isSuccess && item.thumbSrc) {
                html += '<button class="queue-item-btn" data-action="view" data-idx="' + idx + '">View</button>';
            }
            html += '</div>' +
                '<button class="queue-item-dismiss" data-action="dismiss" data-idx="' + idx + '">\u00d7</button>';

            row.innerHTML = html;
            els.historyList.appendChild(row);
        });

        // Event delegation (use onclick to avoid accumulating listeners)
        els.historyList.onclick = handleHistoryClick;
    }

    function handleHistoryClick(e) {
        var target = e.target.closest('[data-action]');
        if (!target) return;
        var action = target.dataset.action;
        var idx = parseInt(target.dataset.idx);

        if (action === 'view') {
            var item = history[idx];
            if (item && item.thumbSrc) {
                // Store for Generate tab to pick up
                localStorage.setItem('sf-view-image', JSON.stringify({
                    src: item.thumbSrc,
                    isVideo: /\.(webp|mp4|gif)$/i.test(item.filename || '')
                }));
                if (typeof switchTab === 'function') {
                    switchTab('generate');
                }
            }
        } else if (action === 'dismiss') {
            history.splice(idx, 1);
            renderHistory();
            saveHistory();
        }
    }

    function formatTimeAgo(timestamp) {
        var diff = Math.floor((Date.now() - timestamp) / 1000);
        if (diff < 60) return 'just now';
        if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
        return Math.floor(diff / 86400) + 'd ago';
    }

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function saveHistory() {
        try {
            localStorage.setItem('sf-queue-history', JSON.stringify(history));
        } catch(e) { /* quota */ }
    }

    function restoreHistory() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-queue-history'));
            if (saved && Array.isArray(saved)) {
                history = saved.slice(0, 50);
            }
        } catch(e) { /* ignore */ }
    }

    function init() {
        if (initialized) return;
        initialized = true;

        buildUI();
        bindEvents();
        restoreHistory();
        renderHistory();
        connectWS();
    }

    return { init: init };
})();
