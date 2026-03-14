/**
 * Shared WebSocket module — single connection used by all tabs.
 *
 * Wraps SFApi's WebSocket with a stable, persistent clientId and a
 * simple pub/sub interface that any tab can subscribe to.
 *
 * Usage:
 *   SerenityWS.on('progress', (data) => { ... });
 *   SerenityWS.on('executed', (data) => { ... });
 *   SerenityWS.on('*', (msg) => { ... });  // wildcard
 *   SerenityWS.off('progress', fn);
 *   SerenityWS.getClientId();
 */

var SerenityWS = (function() {
    'use strict';

    // Persistent client ID across page loads
    var clientId = localStorage.getItem('sf-client-id');
    if (!clientId) {
        clientId = crypto.randomUUID();
        localStorage.setItem('sf-client-id', clientId);
    }

    var socket = null;
    var listeners = {};
    var reconnectTimer = null;
    var connected = false;

    function connect() {
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        if (socket) {
            try { socket.close(); } catch (e) { /* ignore */ }
        }

        var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        var url = protocol + '//' + location.host + '/ws?clientId=' + clientId;
        socket = new WebSocket(url);

        socket.onopen = function() {
            connected = true;
            emit('connected', null);
        };

        socket.onmessage = function(event) {
            if (event.data instanceof Blob) {
                emit('preview', { blob: event.data });
                return;
            }
            try {
                var msg = JSON.parse(event.data);
                emit(msg.type, msg.data);
            } catch (e) {
                // ignore parse errors
            }
        };

        socket.onclose = function() {
            connected = false;
            emit('disconnected', null);
            reconnectTimer = setTimeout(connect, 2000);
        };

        socket.onerror = function() {
            emit('error', null);
        };
    }

    function emit(type, data) {
        var fns = listeners[type] || [];
        for (var i = 0; i < fns.length; i++) {
            try { fns[i](data); } catch (e) { console.error('WS handler error:', e); }
        }
        // Wildcard listeners
        var wild = listeners['*'] || [];
        for (var j = 0; j < wild.length; j++) {
            try { wild[j]({ type: type, data: data }); } catch (e) { /* ignore */ }
        }
    }

    function on(type, fn) {
        if (!listeners[type]) listeners[type] = [];
        listeners[type].push(fn);
    }

    function off(type, fn) {
        if (!listeners[type]) return;
        listeners[type] = listeners[type].filter(function(f) { return f !== fn; });
    }

    function send(data) {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(data));
        }
    }

    function getClientId() { return clientId; }

    function isConnected() { return connected; }

    // Auto-connect on load
    connect();

    return {
        on: on,
        off: off,
        send: send,
        getClientId: getClientId,
        isConnected: isConnected
    };
})();
