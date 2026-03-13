/**
 * Inline widgets for node inputs.
 * Uses Konva shapes with HTML overlay for actual editing.
 */

const WIDGET_HEIGHT = 22;
const WIDGET_PADDING = 4;

class SFWidgetManager {
    constructor() {
        this.activeOverlay = null;
    }

    removeOverlay() {
        if (this.activeOverlay) {
            this.activeOverlay.remove();
            this.activeOverlay = null;
        }
    }

    /**
     * Create a Konva group representing a widget for a node input.
     * Returns { group, getValue, setValue, height }
     */
    createWidget(node, inputName, inputDef, x, y, width) {
        const type = this._detectType(inputDef);

        switch (type) {
            case 'INT':
            case 'FLOAT':
                return this._createNumberWidget(node, inputName, inputDef, x, y, width, type);
            case 'STRING':
                return this._createStringWidget(node, inputName, inputDef, x, y, width);
            case 'BOOLEAN':
                return this._createBooleanWidget(node, inputName, inputDef, x, y, width);
            case 'COMBO':
                return this._createComboWidget(node, inputName, inputDef, x, y, width);
            default:
                return null;
        }
    }

    _detectType(inputDef) {
        if (!inputDef) return null;

        // inputDef is usually [type, config] from object_info
        if (Array.isArray(inputDef)) {
            const t = inputDef[0];
            // Combo: array of options
            if (Array.isArray(t)) return 'COMBO';
            if (typeof t === 'string') {
                const upper = t.toUpperCase();
                if (upper === 'INT' || upper === 'FLOAT' || upper === 'STRING' || upper === 'BOOLEAN') {
                    return upper;
                }
            }
        }
        return null;
    }

    _getDefault(inputDef) {
        if (Array.isArray(inputDef) && inputDef.length > 1 && inputDef[1]) {
            return inputDef[1].default;
        }
        return undefined;
    }

    _getConfig(inputDef) {
        if (Array.isArray(inputDef) && inputDef.length > 1 && inputDef[1]) {
            return inputDef[1];
        }
        return {};
    }

    _createNumberWidget(node, inputName, inputDef, x, y, width, type) {
        const config = this._getConfig(inputDef);
        const defaultVal = config.default !== undefined ? config.default : 0;
        const min = config.min !== undefined ? config.min : -Infinity;
        const max = config.max !== undefined ? config.max : Infinity;
        const step = config.step !== undefined ? config.step : (type === 'INT' ? 1 : 0.01);

        let value = defaultVal;

        const group = new Konva.Group({ x: x, y: y });

        // Background bar
        const bg = new Konva.Rect({
            width: width,
            height: WIDGET_HEIGHT,
            fill: '#1a1a35',
            cornerRadius: 3,
            listening: true,
        });

        // Label
        const label = new Konva.Text({
            x: 6,
            y: 5,
            text: inputName,
            fontSize: 10,
            fill: '#808090',
            listening: false,
        });

        // Value display
        const valueText = new Konva.Text({
            x: width - 6,
            y: 5,
            text: String(value),
            fontSize: 10,
            fill: '#e0e0e0',
            align: 'right',
            listening: false,
        });

        // Right-align value text
        const updateValueText = () => {
            const formatted = type === 'INT' ? String(Math.round(value)) : value.toFixed(2);
            valueText.text(formatted);
            valueText.x(width - 6 - valueText.width());
        };
        updateValueText();

        group.add(bg);
        group.add(label);
        group.add(valueText);

        // Drag to scrub
        let dragStartX = 0;
        let dragStartVal = 0;
        let isDragging = false;

        bg.on('mousedown', (e) => {
            e.cancelBubble = true;
            dragStartX = e.evt.clientX;
            dragStartVal = value;
            isDragging = false;

            const onMove = (me) => {
                const dx = me.clientX - dragStartX;
                if (Math.abs(dx) > 3) isDragging = true;
                if (isDragging) {
                    const sensitivity = me.shiftKey ? 0.1 : 1;
                    const delta = dx * step * sensitivity;
                    value = Math.max(min, Math.min(max, dragStartVal + delta));
                    if (type === 'INT') value = Math.round(value);
                    updateValueText();
                    node.setWidgetValue(inputName, value);
                    node.canvas.nodeLayer.batchDraw();
                }
            };

            const onUp = () => {
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('mouseup', onUp);

                if (!isDragging) {
                    // Click — open overlay input
                    this._openNumberOverlay(node, bg, inputName, value, min, max, step, type, (newVal) => {
                        value = newVal;
                        updateValueText();
                        node.setWidgetValue(inputName, value);
                        node.canvas.nodeLayer.batchDraw();
                    });
                }
            };

            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);
        });

        return {
            group: group,
            getValue: () => value,
            setValue: (v) => {
                value = type === 'INT' ? Math.round(Number(v)) : Number(v);
                value = Math.max(min, Math.min(max, value));
                updateValueText();
            },
            height: WIDGET_HEIGHT,
        };
    }

    _createStringWidget(node, inputName, inputDef, x, y, width) {
        const config = this._getConfig(inputDef);
        const defaultVal = config.default !== undefined ? String(config.default) : '';
        const multiline = config.multiline || false;
        let value = defaultVal;

        const group = new Konva.Group({ x: x, y: y });
        const h = multiline ? WIDGET_HEIGHT * 3 : WIDGET_HEIGHT;

        const bg = new Konva.Rect({
            width: width,
            height: h,
            fill: '#1a1a35',
            cornerRadius: 3,
            listening: true,
        });

        const label = new Konva.Text({
            x: 6,
            y: 3,
            text: inputName,
            fontSize: 9,
            fill: '#606070',
            listening: false,
        });

        const valueText = new Konva.Text({
            x: 6,
            y: multiline ? 14 : 5,
            width: width - 12,
            height: multiline ? h - 16 : h - 6,
            text: value || '(empty)',
            fontSize: 10,
            fill: value ? '#e0e0e0' : '#505060',
            ellipsis: true,
            wrap: multiline ? 'word' : 'none',
            listening: false,
        });

        group.add(bg);
        group.add(label);
        group.add(valueText);

        bg.on('dblclick', (e) => {
            e.cancelBubble = true;
            this._openStringOverlay(node, bg, inputName, value, multiline, width, h, (newVal) => {
                value = newVal;
                valueText.text(value || '(empty)');
                valueText.fill(value ? '#e0e0e0' : '#505060');
                node.setWidgetValue(inputName, value);
                node.canvas.nodeLayer.batchDraw();
            });
        });

        return {
            group: group,
            getValue: () => value,
            setValue: (v) => {
                value = String(v);
                valueText.text(value || '(empty)');
                valueText.fill(value ? '#e0e0e0' : '#505060');
            },
            height: h,
        };
    }

    _createBooleanWidget(node, inputName, inputDef, x, y, width) {
        const config = this._getConfig(inputDef);
        let value = config.default !== undefined ? Boolean(config.default) : false;

        const group = new Konva.Group({ x: x, y: y });

        const bg = new Konva.Rect({
            width: width,
            height: WIDGET_HEIGHT,
            fill: '#1a1a35',
            cornerRadius: 3,
            listening: true,
        });

        const label = new Konva.Text({
            x: 6,
            y: 5,
            text: inputName,
            fontSize: 10,
            fill: '#808090',
            listening: false,
        });

        // Checkbox
        const checkBg = new Konva.Rect({
            x: width - 30,
            y: 4,
            width: 14,
            height: 14,
            fill: value ? '#4a9eff' : '#2a2a4a',
            cornerRadius: 2,
            stroke: '#4a4a6a',
            strokeWidth: 1,
            listening: false,
        });

        const checkMark = new Konva.Text({
            x: width - 27,
            y: 4,
            text: value ? '\u2713' : '',
            fontSize: 12,
            fill: '#fff',
            listening: false,
        });

        group.add(bg);
        group.add(label);
        group.add(checkBg);
        group.add(checkMark);

        bg.on('click', (e) => {
            e.cancelBubble = true;
            value = !value;
            checkBg.fill(value ? '#4a9eff' : '#2a2a4a');
            checkMark.text(value ? '\u2713' : '');
            node.setWidgetValue(inputName, value);
            node.canvas.nodeLayer.batchDraw();
        });

        return {
            group: group,
            getValue: () => value,
            setValue: (v) => {
                value = Boolean(v);
                checkBg.fill(value ? '#4a9eff' : '#2a2a4a');
                checkMark.text(value ? '\u2713' : '');
            },
            height: WIDGET_HEIGHT,
        };
    }

    _createComboWidget(node, inputName, inputDef, x, y, width) {
        const options = Array.isArray(inputDef[0]) ? inputDef[0] : [];
        const config = this._getConfig(inputDef);
        let value = config.default !== undefined ? config.default : (options[0] || '');

        const group = new Konva.Group({ x: x, y: y });

        const bg = new Konva.Rect({
            width: width,
            height: WIDGET_HEIGHT,
            fill: '#1a1a35',
            cornerRadius: 3,
            listening: true,
        });

        const label = new Konva.Text({
            x: 6,
            y: 5,
            text: inputName,
            fontSize: 10,
            fill: '#808090',
            listening: false,
        });

        const valueText = new Konva.Text({
            x: width - 6,
            y: 5,
            text: String(value),
            fontSize: 10,
            fill: '#e0e0e0',
            listening: false,
        });
        valueText.x(width - 6 - valueText.width());

        // Arrow indicator
        const arrow = new Konva.Text({
            x: width - 14,
            y: 5,
            text: '\u25BE',
            fontSize: 10,
            fill: '#808090',
            listening: false,
        });

        group.add(bg);
        group.add(label);
        group.add(valueText);
        group.add(arrow);

        bg.on('click', (e) => {
            e.cancelBubble = true;
            this._openComboOverlay(node, bg, inputName, value, options, width, (newVal) => {
                value = newVal;
                valueText.text(String(value));
                valueText.x(width - 20 - valueText.width());
                node.setWidgetValue(inputName, value);
                node.canvas.nodeLayer.batchDraw();
            });
        });

        return {
            group: group,
            getValue: () => value,
            setValue: (v) => {
                value = v;
                valueText.text(String(value));
                valueText.x(width - 20 - valueText.width());
            },
            height: WIDGET_HEIGHT,
        };
    }

    // --- HTML Overlay helpers ---

    _getOverlayPosition(node, konvaRect) {
        const absPos = konvaRect.getAbsolutePosition();
        const stage = node.canvas.stage;
        const container = stage.container().getBoundingClientRect();
        return {
            x: container.left + absPos.x * stage.scaleX() + stage.x(),
            y: container.top + absPos.y * stage.scaleY() + stage.y(),
        };
    }

    _openNumberOverlay(node, konvaRect, name, currentVal, min, max, step, type, onDone) {
        this.removeOverlay();

        const pos = this._getOverlayPosition(node, konvaRect);
        const scale = node.canvas.stage.scaleX();
        const w = konvaRect.width() * scale;
        const h = konvaRect.height() * scale;

        const div = document.createElement('div');
        div.className = 'widget-overlay';
        div.style.left = pos.x + 'px';
        div.style.top = pos.y + 'px';

        const input = document.createElement('input');
        input.type = 'number';
        input.value = currentVal;
        input.min = min === -Infinity ? '' : min;
        input.max = max === Infinity ? '' : max;
        input.step = step;
        input.style.width = w + 'px';
        input.style.height = h + 'px';

        div.appendChild(input);
        document.body.appendChild(div);
        this.activeOverlay = div;

        input.focus();
        input.select();

        const finish = () => {
            let v = parseFloat(input.value);
            if (isNaN(v)) v = currentVal;
            if (type === 'INT') v = Math.round(v);
            v = Math.max(min, Math.min(max, v));
            this.removeOverlay();
            onDone(v);
        };

        input.addEventListener('blur', finish);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') finish();
            if (e.key === 'Escape') { this.removeOverlay(); }
        });
    }

    _openStringOverlay(node, konvaRect, name, currentVal, multiline, width, height, onDone) {
        this.removeOverlay();

        const pos = this._getOverlayPosition(node, konvaRect);
        const scale = node.canvas.stage.scaleX();
        const w = width * scale;
        const h = height * scale;

        const div = document.createElement('div');
        div.className = 'widget-overlay';
        div.style.left = pos.x + 'px';
        div.style.top = pos.y + 'px';

        let input;
        if (multiline) {
            input = document.createElement('textarea');
            input.style.width = w + 'px';
            input.style.height = h + 'px';
        } else {
            input = document.createElement('input');
            input.type = 'text';
            input.style.width = w + 'px';
            input.style.height = h + 'px';
        }
        input.value = currentVal;

        div.appendChild(input);
        document.body.appendChild(div);
        this.activeOverlay = div;

        input.focus();
        input.select();

        const finish = () => {
            const v = input.value;
            this.removeOverlay();
            onDone(v);
        };

        input.addEventListener('blur', finish);
        if (!multiline) {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') finish();
                if (e.key === 'Escape') { this.removeOverlay(); }
            });
        }
    }

    _openComboOverlay(node, konvaRect, name, currentVal, options, width, onDone) {
        this.removeOverlay();

        const pos = this._getOverlayPosition(node, konvaRect);
        const scale = node.canvas.stage.scaleX();
        const w = width * scale;

        const div = document.createElement('div');
        div.className = 'widget-overlay';
        div.style.left = pos.x + 'px';
        div.style.top = pos.y + 'px';

        const select = document.createElement('select');
        select.style.width = w + 'px';

        options.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt;
            o.text = opt;
            if (opt === currentVal) o.selected = true;
            select.appendChild(o);
        });

        div.appendChild(select);
        document.body.appendChild(div);
        this.activeOverlay = div;

        select.focus();

        const finish = () => {
            const v = select.value;
            this.removeOverlay();
            onDone(v);
        };

        select.addEventListener('change', finish);
        select.addEventListener('blur', finish);
    }
}

// Global widget manager
const sfWidgets = new SFWidgetManager();
