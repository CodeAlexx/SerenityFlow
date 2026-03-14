/**
 * Toolbar: queue, interrupt, clear, save, load. Execution status.
 */
class SFToolbar {
    constructor(canvas, api) {
        this.canvas = canvas;
        this.api = api;

        this.queueBtn = document.getElementById('btn-queue');
        this.interruptBtn = document.getElementById('btn-interrupt');
        this.saveBtn = document.getElementById('btn-save');
        this.loadBtn = document.getElementById('btn-load');
        this.statusIndicator = document.getElementById('status-indicator');
        this.queueCount = document.getElementById('queue-count');
        this.fileInput = document.getElementById('file-input');

        this._setupButtons();
        this._setupKeyboard();
        this._setupApiEvents();
    }

    _setupButtons() {
        this.queueBtn.addEventListener('click', () => this._queuePrompt());
        this.interruptBtn.addEventListener('click', () => this.api.interrupt());
        this.saveBtn.addEventListener('click', () => this._saveWorkflow());
        this.loadBtn.addEventListener('click', () => this.fileInput.click());

        this.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) this._loadWorkflow(file);
            this.fileInput.value = '';
        });

        // Templates are now handled by shell.js dropdown
    }

    _setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
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
                this.fileInput.click();
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

        this.api.on('status', (data) => {
            if (data && data.status) {
                const qr = data.status.exec_info ? data.status.exec_info.queue_remaining : 0;
                this.queueCount.textContent = String(qr);
                this._setStatus(qr > 0 ? 'running' : 'idle');
            }
        });

        this.api.on('executing', (data) => {
            if (data && data.node) {
                // Highlight executing node
                this.canvas.nodes.forEach(n => {
                    if (n._executionState === 'executing') {
                        n.setExecutionState(null);
                    }
                });
                const node = this.canvas.nodes.get(data.node);
                if (node) node.setExecutionState('executing');
            } else if (data && data.node === null) {
                // Execution finished
                this.canvas.nodes.forEach(n => {
                    if (n._executionState === 'executing') {
                        n.setExecutionState('executed');
                    }
                });
            }
        });

        this.api.on('executed', (data) => {
            if (data && data.node) {
                const node = this.canvas.nodes.get(data.node);
                if (node) node.setExecutionState('executed');

                // Check for image outputs
                if (data.output && data.output.images) {
                    data.output.images.forEach(img => {
                        if (sfPreview) {
                            sfPreview.showImage(this.api.viewUrl(img.filename, img.subfolder, img.type));
                        }
                    });
                }

                // Check for video outputs (SaveVideo, SaveAnimatedWEBP)
                if (data.output && data.output.videos) {
                    data.output.videos.forEach(vid => {
                        if (sfPreview) {
                            sfPreview.showVideo(this.api.viewUrl(vid.filename, vid.subfolder, vid.type));
                        }
                    });
                }
            }
        });

        this.api.on('execution_error', (data) => {
            if (data && data.node_id) {
                const node = this.canvas.nodes.get(data.node_id);
                if (node) node.setExecutionState('error');
            }
            this._toast('Execution error: ' + (data.exception_message || 'unknown'), 'error');
        });

        this.api.on('progress', (data) => {
            if (data && data.node) {
                // Could show progress bar on node in the future
            }
        });
    }

    async _queuePrompt() {
        const prompt = serializeWorkflow(this.canvas);
        if (Object.keys(prompt).length === 0) {
            this._toast('No nodes to execute', 'error');
            return;
        }

        try {
            const result = await this.api.queuePrompt(prompt);
            if (result.error) {
                this._toast('Queue error: ' + result.error, 'error');
            }
        } catch (e) {
            this._toast('Failed to queue: ' + e.message, 'error');
        }
    }

    _saveWorkflow() {
        const prompt = serializeWorkflow(this.canvas);

        // Also save node positions for reload
        const workflow = {
            version: 1,
            prompt: prompt,
            nodes: {},
        };
        this.canvas.nodes.forEach((node, id) => {
            workflow.nodes[id] = {
                type: node.nodeType,
                pos: [node.x, node.y],
                widgets_values: { ...node.widgetValues },
            };
        });

        const blob = new Blob([JSON.stringify(workflow, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'workflow.json';
        a.click();
        URL.revokeObjectURL(url);

        this._toast('Workflow saved', 'success');
    }

    _loadWorkflow(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                loadWorkflow(this.canvas, data, this.canvas.nodeInfo);
                this._toast('Workflow loaded');
            } catch (err) {
                this._toast('Failed to load: ' + err.message, 'error');
            }
        };
        reader.readAsText(file);
    }

    /**
     * Load a workflow from a URL (for default/template workflows).
     */
    async loadWorkflowFromUrl(url) {
        try {
            const r = await fetch(url + '?t=' + Date.now(), {cache: 'no-store'});
            if (!r.ok) throw new Error('HTTP ' + r.status);
            const data = await r.json();
            loadWorkflow(this.canvas, data, this.canvas.nodeInfo);
            this._toast('Workflow loaded');
        } catch (err) {
            this._toast('Failed to load workflow: ' + err.message, 'error');
        }
    }

    _setStatus(status) {
        const el = this.statusIndicator;
        el.className = 'status-' + status;
    }

    _toast(msg, variant) {
        const existing = document.querySelector('.toast');
        if (existing) existing.remove();

        const div = document.createElement('div');
        div.className = 'toast' + (variant ? ' toast-' + variant : '');
        div.textContent = msg;
        document.body.appendChild(div);
        setTimeout(() => div.remove(), variant === 'error' ? 5000 : 3000);
    }
}
