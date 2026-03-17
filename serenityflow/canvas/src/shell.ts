/**
 * Shell — Global navigation shell for SerenityFlow.
 * Handles icon rail, tab switching, mode toggle (Simple/Advanced),
 * Lucide icons, Konva resize, templates dropdown, and workflow
 * execution state via SerenityWS.
 */

// --- Cross-file global declarations (only those not defined in other .ts files) ---
// SimpleModeAPI is defined in simple.ts
interface LucideAPI {
    createIcons(opts?: { nameAttr?: string }): void;
}
declare var lucide: LucideAPI;

// Apply persisted settings immediately (Settings module loaded before shell.js)
if (typeof Settings !== 'undefined') {
    Settings.applyTheme(Settings.get('theme'));
    Settings.applyAccentColor(Settings.get('accentColor'));
}

var currentMode = localStorage.getItem('sf-mode') || 'simple';

function setMode(mode: string): void {
    currentMode = mode;
    localStorage.setItem('sf-mode', mode);

    var isAdvanced = mode === 'advanced';
    // Icon rail is always visible (provides navigation in both modes)
    document.getElementById('icon-rail')!.style.display = 'flex';
    document.getElementById('simple-mode-container')!.style.display = isAdvanced ? 'none' : 'flex';
    document.getElementById('content-area')!.style.display = isAdvanced ? 'flex' : 'none';

    // Update toggle button states
    document.getElementById('mode-simple-btn')!.classList.toggle('active', !isAdvanced);
    document.getElementById('mode-advanced-btn')!.classList.toggle('active', isAdvanced);

    // Init Simple mode on first switch
    if (!isAdvanced && typeof SimpleMode !== 'undefined') {
        SimpleMode.init();
    }

    // If switching to advanced, restore the saved tab
    if (isAdvanced) {
        var savedTab = localStorage.getItem('sf-active-tab') || 'generate';
        switchTab(savedTab);
    }
}

function switchTab(tabId: string): void {
    // Update icon rail active state
    document.querySelectorAll('.nav-btn').forEach(function(btn: Element) {
        (btn as HTMLElement).classList.toggle('active', (btn as HTMLElement).dataset.tab === tabId);
    });

    // Show/hide panels
    document.querySelectorAll('.tab-panel').forEach(function(panel: Element) {
        (panel as HTMLElement).style.display = panel.id === 'panel-' + tabId ? 'flex' : 'none';
    });

    // Init Generate tab on first switch
    if (tabId === 'generate' && typeof GenerateTab !== 'undefined') {
        GenerateTab.init();
        // Check for pending image from Queue tab
        var pendingView = localStorage.getItem('sf-view-image');
        if (pendingView) {
            localStorage.removeItem('sf-view-image');
            try {
                var viewData = JSON.parse(pendingView);
                if (viewData.src && GenerateTab.displayResult) {
                    GenerateTab.displayResult(viewData.src, viewData.isVideo);
                }
            } catch(e) {}
        }
    }

    // Init Canvas tab on first switch
    if (tabId === 'canvas' && typeof CanvasTab !== 'undefined') {
        CanvasTab.init();
        requestAnimationFrame(function() { CanvasTab.resize(); });
    }

    // Init Queue tab on first switch
    if (tabId === 'queue' && typeof QueueTab !== 'undefined') {
        QueueTab.init();
    }

    // Init Models tab on first switch
    if (tabId === 'models' && typeof ModelsTab !== 'undefined') {
        ModelsTab.init();
    }

    // Init Settings tab on first switch
    if (tabId === 'settings' && typeof SettingsTab !== 'undefined') {
        SettingsTab.init();
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
function resizeWorkflowStage(): void {
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
function setupTopbarWS(): void {
    if (typeof SerenityWS === 'undefined') return;

    var dot = document.querySelector('.queue-dot') as HTMLElement | null;
    var label = document.querySelector('.queue-label') as HTMLElement | null;
    if (!dot || !label) return;

    SerenityWS.on('connected', function() {
        dot!.className = 'queue-dot idle';
        label!.textContent = 'Idle';
    });

    SerenityWS.on('disconnected', function() {
        dot!.className = 'queue-dot';
        dot!.style.background = 'var(--shell-error)';
        label!.textContent = 'Disconnected';
    });

    SerenityWS.on('status', function(data: WSEventData) {
        if (!data || !data.status) return;
        var qr = data.status.exec_info ? data.status.exec_info.queue_remaining : 0;
        if (qr > 0) {
            dot!.className = 'queue-dot running';
            label!.textContent = 'Running (' + qr + ')';
        } else {
            dot!.className = 'queue-dot idle';
            label!.textContent = 'Idle';
        }
    });
}

/**
 * Setup the workflow execution state — enable/disable Stop button,
 * toggle Queue button running state.
 */
function setupWorkflowExecution(): void {
    if (typeof SerenityWS === 'undefined') return;

    var queueBtn = document.getElementById('btn-queue');
    var stopBtn = document.getElementById('btn-interrupt') as HTMLButtonElement | null;
    if (!queueBtn || !stopBtn) return;

    function setWorkflowRunning(running: boolean): void {
        var queueLabel = queueBtn!.querySelector('span');
        if (queueLabel) {
            queueLabel.textContent = running ? 'Running...' : 'Queue';
        }
        queueBtn!.classList.toggle('running', running);
        stopBtn!.disabled = !running;
    }

    SerenityWS.on('execution_start', function() {
        setWorkflowRunning(true);
    });

    SerenityWS.on('execution_success', function() {
        setWorkflowRunning(false);
        // Clear node highlights after 3s
        setTimeout(function() {
            if (typeof sfCanvas !== 'undefined' && sfCanvas) {
                sfCanvas.nodes.forEach(function(n: SFNode) {
                    n.setExecutionState(null);
                });
            }
        }, 3000);
    });

    SerenityWS.on('execution_error', function() {
        setWorkflowRunning(false);
    });

    // Status-based fallback (when execution finishes without explicit success event)
    SerenityWS.on('status', function(data: WSEventData) {
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
function setupTemplatesDropdown(): void {
    var btn = document.getElementById('btn-templates');
    var dropdown = document.getElementById('templates-dropdown');
    var list = document.getElementById('templates-list');
    if (!btn || !dropdown || !list) return;

    type TemplateEntry = {
        name?: string;
        file?: string;
        url?: string;
    };

    // Known templates as fallback
    const fallbackTemplates: TemplateEntry[] = [
        { name: 'FLUX Text to Image', file: 'flux_t2i.json' },
        { name: 'LTX2 Text to Video', file: 'ltx2_t2v.json' },
    ];

    function renderTemplates(templates: TemplateEntry[]): void {
        list!.innerHTML = '';
        if (!templates || templates.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'wf-template-empty';
            empty.textContent = 'No templates available';
            list!.appendChild(empty);
            return;
        }
        templates.forEach(function(t) {
            var item = document.createElement('div');
            item.className = 'wf-template-item';
            item.textContent = t.name || (t.file ? t.file.replace(/\.json$/i, '').replace(/_/g, ' ') : 'Workflow');
            item.addEventListener('click', function(e) {
                e.stopPropagation();
                dropdown!.classList.add('hidden');
                loadTemplate(t);
            });
            list!.appendChild(item);
        });
    }

    function loadTemplate(template: TemplateEntry): void {
        if (!template) {
            console.error('No template provided');
            return;
        }
        var baseUrl = template.url;
        if (!baseUrl && template.file) {
            baseUrl = 'workflows/' + template.file;
        }
        if (!baseUrl) {
            console.error('Template URL is missing', template);
            return;
        }
        var url = baseUrl + '?t=' + Date.now();
        var nameInput = document.getElementById('workflow-name') as HTMLInputElement | null;
        if (nameInput) {
            var displayName = template.name || (template.file ? template.file.replace(/\.json$/i, '').replace(/_/g, ' ') : '');
            if (displayName) {
                displayName = displayName.charAt(0).toUpperCase() + displayName.slice(1);
                nameInput.value = displayName;
            }
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
        var isOpen = !dropdown!.classList.contains('hidden');
        if (isOpen) {
            dropdown!.classList.add('hidden');
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

        dropdown!.classList.remove('hidden');
    });

    // Close dropdown on outside click
    document.addEventListener('click', function() {
        dropdown!.classList.add('hidden');
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // Init Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    // Attach click handlers to rail buttons
    document.querySelectorAll('.nav-btn').forEach(function(btn: Element) {
        btn.addEventListener('click', function() {
            // If in simple mode, switch to advanced first
            if (currentMode !== 'advanced') {
                setMode('advanced');
            }
            switchTab((btn as HTMLElement).dataset.tab!);
        });
    });

    // Setup shared WS integrations
    setupTopbarWS();
    setupWorkflowExecution();
    setupTemplatesDropdown();

    // Mode toggle buttons
    var simpleModeBtn = document.getElementById('mode-simple-btn');
    var advancedModeBtn = document.getElementById('mode-advanced-btn');
    if (simpleModeBtn) {
        simpleModeBtn.addEventListener('click', function() { setMode('simple'); });
    }
    if (advancedModeBtn) {
        advancedModeBtn.addEventListener('click', function() { setMode('advanced'); });
    }

    // Restore mode (default: advanced).
    // setMode('advanced') internally calls switchTab to restore the saved tab.
    var savedMode = localStorage.getItem('sf-mode') || (typeof Settings !== 'undefined' ? Settings.get('defaultMode') : 'advanced');
    setMode(savedMode);

    // Restore saved extra model directories, then warm object_info cache
    var savedDirs: string[] = [];
    try { savedDirs = JSON.parse(localStorage.getItem('sf-extra-model-dirs') || '[]'); } catch(e) {}
    var dirPromises = savedDirs.map(function(dir: string) {
        return fetch('/folder_paths/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: dir })
        }).catch(function() {});
    });
    Promise.all(dirPromises).then(function() {
        if (typeof ModelUtils !== 'undefined' && ModelUtils.loadObjectInfo) {
            ModelUtils.loadObjectInfo().catch(function() {});
        }
    });

    // ── Keyboard Shortcuts ──
    var tabKeys: Record<string, string> = { '1': 'generate', '2': 'queue', '3': 'canvas', '4': 'models', '5': 'workflows', '6': 'settings' };

    document.addEventListener('keydown', function(e: KeyboardEvent) {
        var target = e.target as HTMLElement;
        var tag = target.tagName;
        var isTyping = (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable);
        var activeTab = localStorage.getItem('sf-active-tab') || 'generate';

        // --- Global: Ctrl+Enter → Generate ---
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (typeof GenerateTab !== 'undefined' && GenerateTab.generate) {
                GenerateTab.generate();
            }
            return;
        }

        // --- Global: Escape → Cancel generation ---
        if (e.key === 'Escape' && !e.ctrlKey && !e.altKey && !e.shiftKey) {
            if (typeof SerenityAPI !== 'undefined' && SerenityAPI.interrupt) {
                SerenityAPI.interrupt();
            }
            return;
        }

        // --- Global: Ctrl+A on generate tab → Select all gallery images ---
        if ((e.ctrlKey || e.metaKey) && e.key === 'a' && activeTab === 'generate' && !isTyping) {
            if (typeof GenerateTab !== 'undefined' && GenerateTab.state && GenerateTab.state.gallery) {
                e.preventDefault();
                var st = GenerateTab.state;
                st.selectedImages = [];
                for (var i = 0; i < st.gallery.length; i++) {
                    st.selectedImages.push(i);
                }
                // Update selection UI via DOM
                var thumbs = document.querySelectorAll('#gen-right-panel .gen-thumb-wrap');
                thumbs.forEach(function(wrap: Element) {
                    var idx = parseInt((wrap as HTMLElement).dataset.galleryIndex!);
                    wrap.classList.toggle('gen-selected', st.selectedImages.indexOf(idx) >= 0);
                });
                var badge = document.getElementById('gen-selection-badge');
                if (badge) {
                    if (st.selectedImages.length > 1) {
                        badge.textContent = st.selectedImages.length + ' selected';
                        badge.classList.add('visible');
                    } else {
                        badge.classList.remove('visible');
                    }
                }
                var bulkBar = document.getElementById('gen-bulk-bar');
                if (bulkBar) {
                    bulkBar.classList.toggle('visible', st.selectedImages.length > 1);
                }
            }
            return;
        }

        // Everything below requires NOT typing in an input
        if (isTyping) return;

        // --- Global: 1-6 → Switch tabs (only in advanced mode) ---
        if (!e.ctrlKey && !e.altKey && !e.metaKey && tabKeys[e.key]) {
            if (currentMode === 'advanced') {
                e.preventDefault();
                switchTab(tabKeys[e.key]);
            }
            return;
        }

        // --- Generate tab only shortcuts ---
        if (activeTab !== 'generate' || typeof GenerateTab === 'undefined') return;

        var leftPanel = document.getElementById('gen-left-panel');
        var rightPanel = document.getElementById('gen-right-panel');
        var st = GenerateTab.state;

        // T or O → Toggle left panel
        if (e.key === 't' || e.key === 'T' || e.key === 'o' || e.key === 'O') {
            if (leftPanel && st) {
                st.leftPanelVisible = !st.leftPanelVisible;
                leftPanel.classList.toggle('gen-panel-hidden', !st.leftPanelVisible);
                var floatBtn = document.getElementById('gen-float-toggle-left');
                if (floatBtn) floatBtn.classList.toggle('active', st.leftPanelVisible);
            }
            return;
        }

        // G → Toggle right panel (gallery)
        if (e.key === 'g' || e.key === 'G') {
            if (rightPanel && st) {
                st.rightPanelVisible = !st.rightPanelVisible;
                rightPanel.classList.toggle('gen-panel-hidden', !st.rightPanelVisible);
                var galleryToggle = document.getElementById('gen-toolbar-toggle-gallery');
                if (galleryToggle) galleryToggle.classList.toggle('active', st.rightPanelVisible);
            }
            return;
        }

        // F → Toggle both panels (full screen preview)
        if (e.key === 'f' || e.key === 'F') {
            if (leftPanel && rightPanel && st) {
                // If either panel is visible, hide both; if both hidden, show both
                var hideAll = st.leftPanelVisible || st.rightPanelVisible;
                st.leftPanelVisible = !hideAll;
                st.rightPanelVisible = !hideAll;
                leftPanel.classList.toggle('gen-panel-hidden', hideAll);
                rightPanel.classList.toggle('gen-panel-hidden', hideAll);
                var floatBtn2 = document.getElementById('gen-float-toggle-left');
                if (floatBtn2) floatBtn2.classList.toggle('active', !hideAll);
                var galleryToggle2 = document.getElementById('gen-toolbar-toggle-gallery');
                if (galleryToggle2) galleryToggle2.classList.toggle('active', !hideAll);
            }
            return;
        }

        // Delete / Backspace → Delete selected gallery images
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (st && st.selectedImages && st.selectedImages.length > 0) {
                e.preventDefault();
                var bulkDelete = document.getElementById('gen-bulk-delete');
                if (bulkDelete) {
                    bulkDelete.click();
                }
            }
            return;
        }

        // . (period) → Toggle star on selected images
        if (e.key === '.') {
            if (st && st.selectedImages && st.selectedImages.length > 0 && st.gallery) {
                st.selectedImages.forEach(function(idx) {
                    if (st.gallery[idx]) {
                        st.gallery[idx].starred = !st.gallery[idx].starred;
                    }
                });
                // Update star icons in DOM
                var thumbs = document.querySelectorAll('#gen-right-panel .gen-thumb-wrap');
                thumbs.forEach(function(wrap: Element) {
                    var idx = parseInt((wrap as HTMLElement).dataset.galleryIndex!);
                    if (st.gallery[idx]) {
                        var starEl = wrap.querySelector('.gen-thumb-star');
                        if (starEl) {
                            var isStarred = st.gallery[idx].starred;
                            starEl.textContent = isStarred ? '\u2605' : '\u2606';
                            starEl.classList.toggle('starred', isStarred);
                        }
                    }
                });
                // Persist
                try {
                    localStorage.setItem('sf-gallery', JSON.stringify(st.gallery));
                } catch(ex) {}
            }
            return;
        }
    });
});
