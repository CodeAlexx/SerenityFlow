/**
 * REST + WebSocket client for SerenityFlow server.
 */
class SFApi {
    constructor(host) {
        host = host || window.location.host;
        const proto = window.location.protocol === 'https:' ? 'https' : 'http';
        this.baseUrl = proto + '://' + host;
        this.clientId = (typeof SerenityWS !== 'undefined') ? SerenityWS.getClientId() : this._generateId();
        this.listeners = {};
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
        // Bridge from shared SerenityWS — no separate socket
        if (typeof SerenityWS === 'undefined') {
            console.warn('SerenityWS not loaded, SFApi WS bridge unavailable');
            return;
        }

        const self = this;

        SerenityWS.on('connected', () => self._emit('connected'));
        SerenityWS.on('disconnected', () => self._emit('disconnected'));
        SerenityWS.on('error', () => self._emit('error'));
        SerenityWS.on('preview', (data) => {
            if (data && data.blob) {
                const url = URL.createObjectURL(data.blob);
                self._emit('preview', { url: url });
            }
        });

        // Forward all message types to SFApi listeners
        SerenityWS.on('status', (d) => self._emit('status', d));
        SerenityWS.on('executing', (d) => self._emit('executing', d));
        SerenityWS.on('executed', (d) => self._emit('executed', d));
        SerenityWS.on('execution_start', (d) => self._emit('execution_start', d));
        SerenityWS.on('execution_error', (d) => self._emit('execution_error', d));
        SerenityWS.on('execution_success', (d) => self._emit('execution_success', d));
        SerenityWS.on('progress', (d) => self._emit('progress', d));

        // If already connected, fire immediately
        if (SerenityWS.isConnected()) {
            self._emit('connected');
        }
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
