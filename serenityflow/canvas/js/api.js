/**
 * SerenityAPI — Shared API helpers for SerenityFlow.
 * All /prompt, /upload, /interrupt calls go through here.
 * Registers jobs with QueueTab when available.
 */
var SerenityAPI = (function() {
    'use strict';

    function postPrompt(workflow, metadata) {
        var meta = metadata || {};
        return fetch('/prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: workflow,
                client_id: SerenityWS.getClientId()
            })
        })
        .then(function(resp) {
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
        .then(function(data) {
            // Register with Queue tab if available
            if (typeof QueueTab !== 'undefined' && QueueTab.registerPending) {
                QueueTab.registerPending({
                    promptId: data.prompt_id,
                    prompt: meta.prompt || '',
                    model: meta.model || '',
                    queuedAt: Date.now(),
                    batchLabel: meta.batchLabel || '',
                    promptData: {
                        workflow: workflow,
                        prompt: meta.prompt || '',
                        model: meta.model || '',
                        width: meta.width || null,
                        height: meta.height || null,
                        seed: meta.seed != null ? meta.seed : null,
                        scheduler: meta.scheduler || null,
                        steps: meta.steps || null,
                        cfg: meta.cfg || null
                    }
                });
            }
            return data;
        });
    }

    function interrupt() {
        return fetch('/interrupt', { method: 'POST' });
    }

    function uploadImage(base64Data, prefix) {
        return fetch('data:image/png;base64,' + base64Data)
            .then(function(r) { return r.blob(); })
            .then(function(blob) {
                var form = new FormData();
                form.append('image', blob, (prefix || 'upload') + '.png');
                form.append('type', 'input');
                return fetch('/upload/image', { method: 'POST', body: form });
            })
            .then(function(resp) {
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.json();
            })
            .then(function(data) { return data.name; });
    }

    function viewUrl(filename, subfolder, type) {
        return '/view?filename=' + encodeURIComponent(filename) +
            '&subfolder=' + encodeURIComponent(subfolder || '') +
            '&type=' + encodeURIComponent(type || 'output');
    }

    return {
        postPrompt: postPrompt,
        interrupt: interrupt,
        uploadImage: uploadImage,
        viewUrl: viewUrl
    };
})();

/**
 * SFApi — Compatibility shim for the graph editor (app.js, toolbar.js,
 * preview.js, sidebar.js). Delegates to SerenityWS and SerenityAPI.
 */
function SFApi() {
    this.connect = function() {};
    this.on = function(type, fn) { SerenityWS.on(type, fn); };
    this.off = function(type, fn) { SerenityWS.off(type, fn); };
    this.interrupt = function() { return SerenityAPI.interrupt(); };
    this.viewUrl = function(filename, subfolder, type) {
        return SerenityAPI.viewUrl(filename, subfolder, type);
    };
    this.getObjectInfo = function() {
        return fetch('/object_info').then(function(r) { return r.json(); });
    };
    this.queuePrompt = function(workflow) {
        return SerenityAPI.postPrompt(workflow);
    };
    this.getClientId = function() { return SerenityWS.getClientId(); };
}
