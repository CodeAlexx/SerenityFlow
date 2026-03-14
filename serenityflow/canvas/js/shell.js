/**
 * Shell — Global navigation shell for SerenityFlow.
 * Handles icon rail, tab switching, Lucide icons, Konva resize,
 * templates dropdown, and workflow execution state via SerenityWS.
 */

function switchTab(tabId) {
    // Update icon rail active state
    document.querySelectorAll('.nav-btn').forEach(function(btn) {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });

    // Show/hide panels
    document.querySelectorAll('.tab-panel').forEach(function(panel) {
        panel.style.display = panel.id === 'panel-' + tabId ? 'flex' : 'none';
    });

    // Init Generate tab on first switch
    if (tabId === 'generate' && typeof GenerateTab !== 'undefined') {
        GenerateTab.init();
    }

    // Init Canvas tab on first switch
    if (tabId === 'canvas' && typeof CanvasTab !== 'undefined') {
        CanvasTab.init();
        requestAnimationFrame(function() { CanvasTab.resize(); });
    }

    // Resize Konva stage when workflows tab becomes visible
    if (tabId === 'workflows') {
        requestAnimationFrame(resizeWorkflowStage);
    }

    // Persist
    localStorage.setItem('sf-active-tab', tabId);
}

/**
 * Resize the Konva stage to fill available space within the workflows panel.
 * Accounts for sidebar, properties panel, and toolbar heights.
 */
function resizeWorkflowStage() {
    if (typeof sfCanvas === 'undefined' || !sfCanvas || !sfCanvas.stage) return;

    var container = document.getElementById('canvas-container');
    if (!container) return;

    // canvas-container has flex:1, fills full width (no sidebar).
    sfCanvas.stage.width(container.offsetWidth);
    sfCanvas.stage.height(container.offsetHeight);
    sfCanvas.stage.batchDraw();
}

window.addEventListener('resize', function() {
    var activeTab = localStorage.getItem('sf-active-tab') || 'generate';
    if (activeTab === 'workflows') {
        resizeWorkflowStage();
    }
    if (activeTab === 'canvas' && typeof CanvasTab !== 'undefined') {
        CanvasTab.resize();
    }
});

/**
 * Update topbar connection indicator from SerenityWS.
 */
function setupTopbarWS() {
    if (typeof SerenityWS === 'undefined') return;

    var dot = document.querySelector('.queue-dot');
    var label = document.querySelector('.queue-label');
    if (!dot || !label) return;

    SerenityWS.on('connected', function() {
        dot.className = 'queue-dot idle';
        label.textContent = 'Idle';
    });

    SerenityWS.on('disconnected', function() {
        dot.className = 'queue-dot';
        dot.style.background = 'var(--shell-error)';
        label.textContent = 'Disconnected';
    });

    SerenityWS.on('status', function(data) {
        if (!data || !data.status) return;
        var qr = data.status.exec_info ? data.status.exec_info.queue_remaining : 0;
        if (qr > 0) {
            dot.className = 'queue-dot running';
            label.textContent = 'Running (' + qr + ')';
        } else {
            dot.className = 'queue-dot idle';
            label.textContent = 'Idle';
        }
    });
}

/**
 * Setup the workflow execution state — enable/disable Stop button,
 * toggle Queue button running state.
 */
function setupWorkflowExecution() {
    if (typeof SerenityWS === 'undefined') return;

    var queueBtn = document.getElementById('btn-queue');
    var stopBtn = document.getElementById('btn-interrupt');
    if (!queueBtn || !stopBtn) return;

    function setWorkflowRunning(running) {
        var queueLabel = queueBtn.querySelector('span');
        if (queueLabel) {
            queueLabel.textContent = running ? 'Running...' : 'Queue';
        }
        queueBtn.classList.toggle('running', running);
        stopBtn.disabled = !running;
    }

    SerenityWS.on('execution_start', function() {
        setWorkflowRunning(true);
    });

    SerenityWS.on('execution_success', function() {
        setWorkflowRunning(false);
        // Clear node highlights after 3s
        setTimeout(function() {
            if (typeof sfCanvas !== 'undefined' && sfCanvas) {
                sfCanvas.nodes.forEach(function(n) {
                    n.setExecutionState(null);
                });
            }
        }, 3000);
    });

    SerenityWS.on('execution_error', function() {
        setWorkflowRunning(false);
    });

    // Status-based fallback (when execution finishes without explicit success event)
    SerenityWS.on('status', function(data) {
        if (!data || !data.status) return;
        var qr = data.status.exec_info ? data.status.exec_info.queue_remaining : 0;
        if (qr === 0) {
            setWorkflowRunning(false);
        }
    });
}

/**
 * Setup Templates dropdown.
 */
function setupTemplatesDropdown() {
    var btn = document.getElementById('btn-templates');
    var dropdown = document.getElementById('templates-dropdown');
    var list = document.getElementById('templates-list');
    if (!btn || !dropdown || !list) return;

    // Known templates as fallback
    var fallbackTemplates = [
        { name: 'FLUX Text to Image', file: 'flux_t2i.json' },
        { name: 'LTX2 Text to Video', file: 'ltx2_t2v.json' }
    ];

    function renderTemplates(templates) {
        list.innerHTML = '';
        if (!templates || templates.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'wf-template-empty';
            empty.textContent = 'No templates available';
            list.appendChild(empty);
            return;
        }
        templates.forEach(function(t) {
            var item = document.createElement('div');
            item.className = 'wf-template-item';
            item.textContent = t.name;
            item.addEventListener('click', function(e) {
                e.stopPropagation();
                dropdown.classList.add('hidden');
                loadTemplate(t.file);
            });
            list.appendChild(item);
        });
    }

    function loadTemplate(filename) {
        var url = 'workflows/' + filename + '?t=' + Date.now();
        // Update workflow name from template filename
        var nameInput = document.getElementById('workflow-name');
        if (nameInput) {
            var name = filename.replace(/\.json$/i, '').replace(/_/g, ' ');
            name = name.charAt(0).toUpperCase() + name.slice(1);
            nameInput.value = name;
        }
        if (typeof sfToolbar !== 'undefined' && sfToolbar) {
            sfToolbar.loadWorkflowFromUrl(url);
        } else {
            fetch(url, { cache: 'no-store' })
                .then(function(r) {
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    return r.json();
                })
                .then(function(data) {
                    if (typeof loadWorkflow !== 'undefined' && typeof sfCanvas !== 'undefined') {
                        loadWorkflow(sfCanvas, data, sfCanvas.nodeInfo);
                    }
                })
                .catch(function(err) {
                    console.error('Failed to load template:', err);
                });
        }
    }

    // Toggle dropdown
    btn.addEventListener('click', function(e) {
        e.stopPropagation();
        var isOpen = !dropdown.classList.contains('hidden');
        if (isOpen) {
            dropdown.classList.add('hidden');
            return;
        }

        // Try to fetch template list from server, fall back to hardcoded
        fetch('/templates')
            .then(function(r) {
                if (!r.ok) throw new Error('no endpoint');
                return r.json();
            })
            .then(function(templates) {
                if (templates && templates.length > 0) {
                    renderTemplates(templates);
                } else {
                    renderTemplates(fallbackTemplates);
                }
            })
            .catch(function() {
                renderTemplates(fallbackTemplates);
            });

        dropdown.classList.remove('hidden');
    });

    // Close dropdown on outside click
    document.addEventListener('click', function() {
        dropdown.classList.add('hidden');
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // Init Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    // Attach click handlers to rail buttons
    document.querySelectorAll('.nav-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            switchTab(btn.dataset.tab);
        });
    });

    // Setup shared WS integrations
    setupTopbarWS();
    setupWorkflowExecution();
    setupTemplatesDropdown();

    // Restore saved tab (default: generate)
    // Note: panel-workflows starts visible so app.js can init Konva with correct
    // container dimensions. app.js DOMContentLoaded runs before this handler
    // (loaded earlier in script order). Now we switch to the user's saved tab.
    var savedTab = localStorage.getItem('sf-active-tab') || 'generate';
    switchTab(savedTab);
});
