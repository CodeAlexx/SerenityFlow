/**
 * Toolbar: queue, interrupt, clear, save, load. Execution status.
 */

// Cross-file globals (script-mode, no import/export)
// sfCanvas, sfPreview, sfHistory declared in app.ts
// workflowMeta declared in workflow.ts
// serializeWorkflow is defined in workflow.ts

interface SFToolbarApi {
    on(event: string, handler: (data: WSEventData) => void): void;
    interrupt(): void;
    queuePrompt(prompt: ComfyPrompt): Promise<{ error?: string }>;
    viewUrl(filename: string, subfolder: string, type: string): string;
}

interface SFToolbarNote {
    element: HTMLDivElement;
    id: string;
}

class SFToolbar {
    canvas: SFCanvas;
    api: SFToolbarApi;

    queueBtn: HTMLElement | null;
    interruptBtn: HTMLElement | null;
    saveBtn: HTMLElement | null;
    loadBtn: HTMLElement | null;
    statusIndicator: HTMLElement | null;
    queueCount: HTMLElement | null;
    fileInput: HTMLInputElement | null;

    _badgeContainer: HTMLDivElement | null;
    _badges: Map<string, HTMLDivElement>;
    _notes: SFToolbarNote[];
    _nextNoteId: number;
    _isRunning: boolean;

    constructor(canvas: SFCanvas, api: SFToolbarApi) {
        this.canvas = canvas;
        this.api = api;

        this.queueBtn = document.getElementById('btn-queue');
        this.interruptBtn = document.getElementById('btn-interrupt');
        this.saveBtn = document.getElementById('btn-save');
        this.loadBtn = document.getElementById('btn-load');
        this.statusIndicator = document.getElementById('status-indicator');
        this.queueCount = document.getElementById('queue-count');
        this.fileInput = document.getElementById('file-input') as HTMLInputElement | null;

        this._badgeContainer = null; // lazy-created div for status badges
        this._badges = new Map();    // nodeId -> badge element
        this._notes = [];            // {element, id}
        this._nextNoteId = 1;
        this._isRunning = false;

        this._setupButtons();
        this._setupKeyboard();
        this._setupApiEvents();
        this._patchCanvasEdgeAnimation();
        this._setupNoteButton();
        this._setupBadgeReposition();
    }

    _setupButtons() {
        this.queueBtn!.addEventListener('click', () => this._queuePrompt());
        this.interruptBtn!.addEventListener('click', () => this.api.interrupt());
        this.saveBtn!.addEventListener('click', () => this._saveWorkflow());
        this.loadBtn!.addEventListener('click', () => this.fileInput!.click());

        this.fileInput!.addEventListener('change', (e: Event) => {
            const target = e.target as HTMLInputElement;
            const file = target.files![0];
            if (file) this._loadWorkflow(file);
            this.fileInput!.value = '';
        });

        // Templates are now handled by shell.js dropdown
    }

    _setupKeyboard() {
        document.addEventListener('keydown', (e: KeyboardEvent) => {
            const target = e.target as HTMLElement;
            if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable) {
                return;
            }

            // Ctrl+Enter - queue
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this._queuePrompt();
            }

            // Space - quick queue
            if (e.key === ' ' && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
                e.preventDefault();
                this._queuePrompt();
            }

            // Ctrl+S - save
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this._saveWorkflow();
            }

            // Ctrl+O - load
            if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
                e.preventDefault();
                this.fileInput!.click();
            }
        });
    }

    _setupApiEvents() {
        this.api.on('connected', () => {
            this._setStatus('idle');
        });

        this.api.on('disconnected', () => {
            this._setStatus('error');
        });

        this.api.on('status', (data: WSEventData) => {
            if (data && data.status) {
                const qr = data.status.exec_info ? data.status.exec_info.queue_remaining : 0;
                this.queueCount!.textContent = String(qr);
                const wasRunning = this._isRunning;
                this._isRunning = qr > 0;
                this._setStatus(qr > 0 ? 'running' : 'idle');

                // Start edge animation when execution begins
                if (this._isRunning && !wasRunning) {
                    this._setEdgesAnimated(true);
                }
                // Stop edge animation when queue drains
                if (!this._isRunning && wasRunning) {
                    this._setEdgesAnimated(false);
                }
            }
        });

        this.api.on('executing', (data: WSEventData) => {
            if (data && data.node) {
                // Highlight executing node, clear previous executing badges
                this.canvas.nodes.forEach((n: SFNode, id: string) => {
                    if (n._executionState === 'executing') {
                        n.setExecutionState(null);
                        this._updateBadge(id, null);
                    }
                });
                const node = this.canvas.nodes.get(data.node);
                if (node) {
                    node.setExecutionState('executing');
                    this._updateBadge(data.node, 'executing');
                }
            } else if (data && data.node === null) {
                // Execution finished
                this.canvas.nodes.forEach((n: SFNode, id: string) => {
                    if (n._executionState === 'executing') {
                        n.setExecutionState('executed');
                        this._updateBadge(id, 'completed');
                    }
                });
                this._setEdgesAnimated(false);
            }
        });

        this.api.on('executed', (data: WSEventData) => {
            if (data && data.node) {
                const node = this.canvas.nodes.get(data.node);
                if (node) node.setExecutionState('executed');
                this._updateBadge(data.node, 'completed');

                // Check for image outputs
                if (data.output && data.output.images) {
                    data.output.images.forEach((img: ComfyOutputFile) => {
                        if (sfPreview) {
                            sfPreview.showImage(this.api.viewUrl(img.filename, img.subfolder, img.type));
                        }
                    });
                }

                // Check for video outputs (SaveVideo, SaveAnimatedWEBP)
                if (data.output && data.output.videos) {
                    data.output.videos.forEach((vid: ComfyOutputFile) => {
                        if (sfPreview) {
                            sfPreview.showVideo(this.api.viewUrl(vid.filename, vid.subfolder, vid.type));
                        }
                    });
                }
            }
        });

        this.api.on('execution_error', (data: WSEventData) => {
            if (data && data.node_id) {
                const node = this.canvas.nodes.get(data.node_id);
                if (node) node.setExecutionState('error');
                this._updateBadge(data.node_id, 'error');
            }
            this._setEdgesAnimated(false);
            this._toast('Execution error: ' + (data.exception_message || 'unknown'), 'error');
        });

        this.api.on('progress', (data: WSEventData) => {
            if (data && data.node) {
                // Could show progress bar on node in the future
            }
        });
    }

    async _queuePrompt() {
        const { prompt } = serializeWorkflow(this.canvas);
        if (Object.keys(prompt).length === 0) {
            this._toast('No nodes to execute', 'error');
            return;
        }

        try {
            const result = await this.api.queuePrompt(prompt);
            if (result.error) {
                this._toast('Queue error: ' + result.error, 'error');
            }
        } catch (e: unknown) {
            this._toast('Failed to queue: ' + (e instanceof Error ? e.message : String(e)), 'error');
        }
    }

    _saveWorkflow() {
        const { prompt, nodePositions } = serializeWorkflow(this.canvas);

        const workflow = {
            version: 2,
            prompt: prompt,
            nodes: nodePositions,
            metadata: typeof workflowMeta !== 'undefined' ? { ...workflowMeta } : {},
        };

        const filename = (workflowMeta && workflowMeta.name ? workflowMeta.name.replace(/[^a-z0-9_-]/gi, '_') : 'workflow') + '.json';
        const blob = new Blob([JSON.stringify(workflow, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);

        this._toast('Workflow saved', 'success');
    }

    _loadWorkflow(file: File) {
        const reader = new FileReader();
        reader.onload = (e: ProgressEvent<FileReader>) => {
            try {
                const data = JSON.parse(e.target!.result as string);
                loadWorkflow(this.canvas, data, this.canvas.nodeInfo);
                this._toast('Workflow loaded', 'success');
            } catch (err: unknown) {
                this._toast('Failed to load: ' + (err instanceof Error ? err.message : String(err)), 'error');
            }
        };
        reader.readAsText(file);
    }

    /**
     * Load a workflow from a URL (for default/template workflows).
     */
    async loadWorkflowFromUrl(url: string) {
        try {
            const r = await fetch(url + '?t=' + Date.now(), {cache: 'no-store'});
            if (!r.ok) throw new Error('HTTP ' + r.status);
            const data = await r.json();
            loadWorkflow(this.canvas, data, this.canvas.nodeInfo);
            this._toast('Workflow loaded', 'success');
        } catch (err: unknown) {
            this._toast('Failed to load workflow: ' + (err instanceof Error ? err.message : String(err)), 'error');
        }
    }

    _setStatus(status: string) {
        const el = this.statusIndicator!;
        el.className = 'status-' + status;
    }

    _toast(msg: string, variant?: string) {
        const existing = document.querySelector('.toast');
        if (existing) existing.remove();

        const div = document.createElement('div');
        div.className = 'toast' + (variant ? ' toast-' + variant : '');
        div.textContent = msg;
        document.body.appendChild(div);
        setTimeout(() => div.remove(), variant === 'error' ? 5000 : 3000);
    }

    // -- Edge animation during execution --

    _patchCanvasEdgeAnimation() {
        var self = this;
        // Monkey-patch setEdgesAnimated onto sfCanvas once available
        var tryPatch = function() {
            if (typeof sfCanvas !== 'undefined' && sfCanvas && !sfCanvas.setEdgesAnimated) {
                sfCanvas.setEdgesAnimated = function(animated: boolean) {
                    this.connections.forEach(function(conn: SFConnection) {
                        if (conn.setAnimated) conn.setAnimated(animated);
                    });
                };
            }
        };
        tryPatch();
        // Also try after a tick in case sfCanvas is set later
        setTimeout(tryPatch, 0);
    }

    _setEdgesAnimated(animated: boolean) {
        if (typeof sfCanvas !== 'undefined' && sfCanvas && sfCanvas.setEdgesAnimated) {
            sfCanvas.setEdgesAnimated(animated);
        }
    }

    // -- Node status badges --

    _ensureBadgeContainer(): HTMLDivElement | null {
        if (this._badgeContainer) return this._badgeContainer;
        var container = document.getElementById('canvas-container');
        if (!container) return null;
        var div = document.createElement('div');
        div.style.position = 'absolute';
        div.style.top = '0';
        div.style.left = '0';
        div.style.width = '100%';
        div.style.height = '100%';
        div.style.pointerEvents = 'none';
        div.style.overflow = 'hidden';
        div.style.zIndex = '200';
        container.appendChild(div);
        this._badgeContainer = div;
        return div;
    }

    _updateBadge(nodeId: string, state: string | null) {
        var container = this._ensureBadgeContainer();
        if (!container) return;

        // Remove existing badge for this node
        var existing = this._badges.get(nodeId);
        if (existing) {
            existing.remove();
            this._badges.delete(nodeId);
        }

        if (!state) return;

        var node = this.canvas.nodes.get(nodeId);
        if (!node) return;

        var badge = document.createElement('div');
        badge.className = 'node-status-badge ' + state;

        if (state === 'executing') {
            badge.textContent = '\u25CF'; // filled circle
        } else if (state === 'completed') {
            badge.textContent = '\u2713'; // check
        } else if (state === 'error') {
            badge.textContent = '\u2717'; // X
        }

        container.appendChild(badge);
        this._badges.set(nodeId, badge);
        this._positionBadge(nodeId, badge);

        // Fade completed badges after 2s
        if (state === 'completed') {
            setTimeout(function() {
                badge.style.opacity = '0';
                setTimeout(function() { badge.remove(); }, 300);
            }, 2000);
        }
    }

    _positionBadge(nodeId: string, badge: HTMLDivElement) {
        var node = this.canvas.nodes.get(nodeId);
        if (!node || !node.group) return;

        var stage = this.canvas.stage;
        var stageBox = stage.container().getBoundingClientRect();
        var transform = stage.getAbsoluteTransform();

        // Node position in screen coordinates
        var pos = transform.point({ x: node.x, y: node.y });
        var nodeW = (node.width || 200) * this.canvas.scale;

        badge.style.left = (pos.x + nodeW - 6) + 'px';
        badge.style.top = (pos.y - 6) + 'px';
    }

    _repositionAllBadges() {
        var self = this;
        this._badges.forEach(function(badge: HTMLDivElement, nodeId: string) {
            self._positionBadge(nodeId, badge);
        });
    }

    _setupBadgeReposition() {
        var self = this;
        // Reposition badges on pan, zoom, and node drag
        this.canvas.stage.on('dragmove.badges', function() {
            self._repositionAllBadges();
        });
        this.canvas.stage.on('scaleChange.badges', function() {
            self._repositionAllBadges();
        });
        this.canvas.nodeLayer.on('dragmove.badges', function() {
            self._repositionAllBadges();
        });
    }

    // -- Notes nodes --

    _setupNoteButton() {
        // Insert a "Note" button in the toolbar-center area
        var center = document.querySelector('#toolbar .toolbar-center');
        if (!center) return;

        var btn = document.createElement('button');
        btn.id = 'btn-add-note';
        btn.className = 'wf-btn-ghost';
        btn.title = 'Add Note';
        btn.innerHTML = '<span>+ Note</span>';
        center.appendChild(btn);

        var self = this;
        btn.addEventListener('click', function() {
            self._addNote();
        });
    }

    _addNote() {
        var canvasContainer = document.getElementById('canvas-container');
        if (!canvasContainer) return;

        var noteId = 'note-' + this._nextNoteId++;

        // Place in center of visible canvas area
        var rect = canvasContainer.getBoundingClientRect();
        var startX = rect.width / 2 - 80;
        var startY = rect.height / 2 - 40;

        var note = document.createElement('div');
        note.className = 'canvas-note';
        note.id = noteId;
        note.style.left = startX + 'px';
        note.style.top = startY + 'px';

        var textarea = document.createElement('textarea');
        textarea.placeholder = 'Type a note...';
        textarea.rows = 3;
        note.appendChild(textarea);

        // Close button
        var closeBtn = document.createElement('button');
        closeBtn.textContent = '\u00d7';
        closeBtn.style.cssText = 'position:absolute;top:2px;right:4px;background:none;border:none;color:#ffd866;cursor:pointer;font-size:14px;line-height:1;padding:2px 4px;opacity:0.6;';
        closeBtn.addEventListener('mouseenter', function() { closeBtn.style.opacity = '1'; });
        closeBtn.addEventListener('mouseleave', function() { closeBtn.style.opacity = '0.6'; });
        closeBtn.addEventListener('click', function() {
            note.remove();
        });
        note.appendChild(closeBtn);

        // Make draggable
        var dragging = false, dragOffX = 0, dragOffY = 0;
        note.addEventListener('mousedown', function(e: MouseEvent) {
            if (e.target === textarea) return;
            dragging = true;
            dragOffX = e.clientX - note.offsetLeft;
            dragOffY = e.clientY - note.offsetTop;
            e.preventDefault();
        });
        var onMouseMove = function(e: MouseEvent) {
            if (!dragging) return;
            note.style.left = (e.clientX - dragOffX) + 'px';
            note.style.top = (e.clientY - dragOffY) + 'px';
        };
        var onMouseUp = function() {
            dragging = false;
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);

        // Clean up listeners when note is removed
        var self = this;
        var origCloseHandler = closeBtn.onclick;
        closeBtn.addEventListener('click', function() {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            self._notes = self._notes.filter(function(n: SFToolbarNote) { return n.id !== noteId; });
        });

        canvasContainer.appendChild(note);
        textarea.focus();

        this._notes.push({ element: note, id: noteId });
    }
}
