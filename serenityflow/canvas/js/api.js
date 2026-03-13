/**
 * REST + WebSocket client for SerenityFlow server.
 */
class SFApi {
    constructor(host) {
        host = host || window.location.host;
        const proto = window.location.protocol === 'https:' ? 'https' : 'http';
        const wsproto = proto === 'https' ? 'wss' : 'ws';
        this.baseUrl = proto + '://' + host;
        this.wsUrl = wsproto + '://' + host + '/ws';
        this.ws = null;
        this.clientId = this._generateId();
        this.listeners = {};
        this._reconnectTimer = null;
    }

    // --- REST ---

    async getObjectInfo() {
        const r = await fetch(this.baseUrl + '/object_info');
        if (!r.ok) throw new Error('Failed to fetch object_info: ' + r.status);
        return r.json();
    }

    async getSystemStats() {
        const r = await fetch(this.baseUrl + '/system_stats');
        return r.json();
    }

    async queuePrompt(prompt) {
        const r = await fetch(this.baseUrl + '/prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                client_id: this.clientId,
                extra_data: { client_id: this.clientId },
            }),
        });
        return r.json();
    }

    async interrupt() {
        await fetch(this.baseUrl + '/interrupt', { method: 'POST' });
    }

    async clearQueue() {
        await fetch(this.baseUrl + '/queue', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ clear: true }),
        });
    }

    async getQueue() {
        const r = await fetch(this.baseUrl + '/queue');
        return r.json();
    }

    async getHistory() {
        const r = await fetch(this.baseUrl + '/history');
        return r.json();
    }

    async getModels(folder) {
        const r = await fetch(this.baseUrl + '/models/' + encodeURIComponent(folder));
        return r.json();
    }

    async uploadImage(file, type) {
        type = type || 'input';
        const form = new FormData();
        form.append('image', file);
        form.append('type', type);
        const r = await fetch(this.baseUrl + '/upload/image', {
            method: 'POST',
            body: form,
        });
        return r.json();
    }

    viewUrl(filename, subfolder, type) {
        subfolder = subfolder || '';
        type = type || 'output';
        return this.baseUrl + '/view?filename=' + encodeURIComponent(filename) +
            '&subfolder=' + encodeURIComponent(subfolder) +
            '&type=' + encodeURIComponent(type);
    }

    // --- WebSocket ---

    connect() {
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
        if (this.ws) {
            try { this.ws.close(); } catch (e) { /* ignore */ }
        }

        this.ws = new WebSocket(this.wsUrl + '?clientId=' + this.clientId);

        this.ws.onopen = () => {
            this._emit('connected');
        };

        this.ws.onmessage = (event) => {
            if (event.data instanceof Blob) {
                this._handleBinaryPreview(event.data);
                return;
            }
            try {
                const msg = JSON.parse(event.data);
                this._emit(msg.type, msg.data);
            } catch (e) {
                console.warn('WS parse error:', e);
            }
        };

        this.ws.onclose = () => {
            this._emit('disconnected');
            this._reconnectTimer = setTimeout(() => this.connect(), 2000);
        };

        this.ws.onerror = () => {
            this._emit('error');
        };
    }

    on(event, callback) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(callback);
    }

    off(event, callback) {
        if (!this.listeners[event]) return;
        this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }

    _emit(event, data) {
        (this.listeners[event] || []).forEach(cb => {
            try { cb(data); } catch (e) { console.error('Event handler error:', e); }
        });
    }

    async _handleBinaryPreview(blob) {
        const url = URL.createObjectURL(blob);
        this._emit('preview', { url: url });
    }

    _generateId() {
        return 'sf_' + Math.random().toString(36).substr(2, 9);
    }
}
