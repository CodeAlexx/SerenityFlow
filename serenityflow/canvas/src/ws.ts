/**
 * SerenityWS — Shared WebSocket client for SerenityFlow.
 * Singleton pub/sub with exponential backoff reconnect.
 */

type WSListener = (data: WSEventData) => void;

interface SerenityWSAPI {
    on(type: string, fn: WSListener): void;
    off(type: string, fn: WSListener): void;
    send(data: unknown): void;
    getClientId(): string;
    isConnected(): boolean;
}

var SerenityWS: SerenityWSAPI = (function(): SerenityWSAPI {
    'use strict';

    var socket: WebSocket | null = null;
    var listeners: Record<string, WSListener[]> = {};
    var connected: boolean = false;
    var reconnectAttempts: number = 0;
    var reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    var MAX_RECONNECT_DELAY: number = 30000;
    var BASE_DELAY: number = 1000;

    // Persistent client ID
    var clientId: string = localStorage.getItem('sf-client-id') || '';
    if (!clientId) {
        clientId = crypto.randomUUID ? crypto.randomUUID() :
            'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c: string): string {
                var r: number = Math.random() * 16 | 0;
                return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
            });
        localStorage.setItem('sf-client-id', clientId);
    }

    function getReconnectDelay(): number {
        var exp: number = Math.min(BASE_DELAY * Math.pow(2, reconnectAttempts), MAX_RECONNECT_DELAY);
        var jitter: number = Math.random() * 0.3 * exp;
        return Math.round(exp + jitter);
    }

    function connect(): void {
        if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
            return;
        }
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }

        var host: string = location.host || 'localhost:8188';
        var protocol: string = location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(protocol + '//' + host + '/ws?clientId=' + clientId);

        socket.onopen = function(): void {
            connected = true;
            reconnectAttempts = 0;
            emit('connected', {});
        };

        socket.onmessage = function(event: MessageEvent): void {
            if (event.data instanceof Blob) {
                emit('preview', { blob: event.data });
                return;
            }
            try {
                var msg = JSON.parse(event.data) as Record<string, unknown>;
                var type = msg.type as string;
                var data: WSEventData = msg.data || msg;
                emit(type, data);
            } catch (e) {
                // Ignore parse errors
            }
        };

        socket.onclose = function(event: CloseEvent): void {
            connected = false;
            emit('disconnected', {});
            // Only reconnect on abnormal close
            if (event.code !== 1000) {
                var delay: number = getReconnectDelay();
                reconnectAttempts++;
                reconnectTimer = setTimeout(connect, delay);
            }
        };

        socket.onerror = function(): void {
            // onclose will fire after this
        };
    }

    function emit(type: string, data: WSEventData): void {
        var handlers: WSListener[] | undefined = listeners[type];
        if (handlers) {
            handlers.forEach(function(fn: WSListener): void {
                try { fn(data); } catch (e) { console.error('WS listener error:', e); }
            });
        }
        // Wildcard listeners
        var wildcard: WSListener[] | undefined = listeners['*'];
        if (wildcard && type !== '*') {
            wildcard.forEach(function(fn: WSListener): void {
                try { fn({ type: type, data: data }); } catch (e) {}
            });
        }
    }

    function on(type: string, fn: WSListener): void {
        if (!listeners[type]) listeners[type] = [];
        listeners[type].push(fn);
    }

    function off(type: string, fn: WSListener): void {
        if (!listeners[type]) return;
        listeners[type] = listeners[type].filter(function(f: WSListener): boolean { return f !== fn; });
    }

    function send(data: unknown): void {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(data));
        }
    }

    function getClientId(): string { return clientId; }
    function isConnected(): boolean { return connected; }

    // Page visibility — reconnect when tab becomes visible
    document.addEventListener('visibilitychange', function(): void {
        if (document.visibilityState === 'visible') {
            if (!socket || socket.readyState === WebSocket.CLOSED) {
                reconnectAttempts = 0;
                connect();
            }
        }
    });

    // Auto-connect
    connect();

    return {
        on: on,
        off: off,
        send: send,
        getClientId: getClientId,
        isConnected: isConnected
    };
})();
