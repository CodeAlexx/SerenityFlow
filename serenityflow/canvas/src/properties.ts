/**
 * Properties panel for selected node.
 */
class SFProperties {
    canvas: SFCanvas;
    panel: HTMLElement;
    title: HTMLElement;
    content: HTMLElement;
    closeBtn: HTMLElement;
    currentNodeId: string | null;

    constructor(canvas: SFCanvas) {
        this.canvas = canvas;
        this.panel = document.getElementById('properties-panel')!;
        this.title = document.getElementById('properties-title')!;
        this.content = document.getElementById('properties-content')!;
        this.closeBtn = document.getElementById('properties-close')!;
        this.currentNodeId = null;

        this.closeBtn.addEventListener('click', () => this.hide());
    }

    show(nodeId: string | number): void {
        const id = String(nodeId);
        const node = this.canvas.nodes.get(id);
        if (!node) return;

        this.currentNodeId = id;
        this.title.textContent = node.info.display_name || node.nodeType;
        this.content.innerHTML = '';

        // Node info group
        const infoGroup = this._createGroup('Node Info');
        this._addRow(infoGroup, 'ID', node.id);
        this._addRow(infoGroup, 'Type', node.nodeType);
        this._addRow(infoGroup, 'Category', node.info.category || 'none');

        // Widget values group
        if (Object.keys(node.widgetValues).length > 0) {
            const valGroup = this._createGroup('Values');

            for (const [name, value] of Object.entries(node.widgetValues)) {
                const inp = node.inputs.find((i: SFNodeInput) => i.name === name);
                if (!inp) continue;

                const type = inp.type;

                if (type === 'STRING') {
                    const config = Array.isArray(inp.config) && inp.config[1] ? inp.config[1] : {};
                    if (config.multiline) {
                        this._addTextarea(valGroup, name, value as string, (v) => {
                            node.setWidgetValue(name, v);
                            if (node.widgets[name]) node.widgets[name].setValue(v);
                            this.canvas.nodeLayer.batchDraw();
                        });
                    } else {
                        this._addInput(valGroup, name, value as string | number, 'text', (v) => {
                            node.setWidgetValue(name, v);
                            if (node.widgets[name]) node.widgets[name].setValue(v);
                            this.canvas.nodeLayer.batchDraw();
                        });
                    }
                } else if (type === 'INT' || type === 'FLOAT') {
                    this._addInput(valGroup, name, value as string | number, 'number', (v) => {
                        const num = type === 'INT' ? parseInt(v) : parseFloat(v);
                        if (!isNaN(num)) {
                            node.setWidgetValue(name, num);
                            if (node.widgets[name]) node.widgets[name].setValue(num);
                            this.canvas.nodeLayer.batchDraw();
                        }
                    });
                } else if (type === 'BOOLEAN') {
                    this._addCheckbox(valGroup, name, value as boolean, (v) => {
                        node.setWidgetValue(name, v);
                        if (node.widgets[name]) node.widgets[name].setValue(v);
                        this.canvas.nodeLayer.batchDraw();
                    });
                } else if (type === 'COMBO') {
                    const options = Array.isArray(inp.config) && Array.isArray(inp.config[0]) ? inp.config[0] : [];
                    this._addSelect(valGroup, name, value as string, options, (v) => {
                        node.setWidgetValue(name, v);
                        if (node.widgets[name]) node.widgets[name].setValue(v);
                        this.canvas.nodeLayer.batchDraw();
                    });
                }
            }
        }

        this.panel.classList.remove('hidden');
    }

    hide(): void {
        this.panel.classList.add('hidden');
        this.currentNodeId = null;
    }

    _createGroup(title: string): HTMLDivElement {
        const div = document.createElement('div');
        div.className = 'prop-group';

        const t = document.createElement('div');
        t.className = 'prop-group-title';
        t.textContent = title;
        div.appendChild(t);

        this.content.appendChild(div);
        return div;
    }

    _addRow(parent: HTMLElement, label: string, value: unknown): void {
        const row = document.createElement('div');
        row.className = 'prop-row';

        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;

        const v = document.createElement('span');
        v.className = 'prop-value';
        v.textContent = String(value);
        v.style.color = 'var(--text-secondary)';
        v.style.fontSize = '12px';

        row.appendChild(l);
        row.appendChild(v);
        parent.appendChild(row);
    }

    _addInput(parent: HTMLElement, label: string, value: string | number, type: string, onChange: (v: string) => void): void {
        const row = document.createElement('div');
        row.className = 'prop-row';

        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;

        const v = document.createElement('div');
        v.className = 'prop-value';

        const input = document.createElement('input');
        input.className = 'prop-input';
        input.type = type;
        input.value = String(value);
        input.addEventListener('change', () => onChange(input.value));

        v.appendChild(input);
        row.appendChild(l);
        row.appendChild(v);
        parent.appendChild(row);
    }

    _addTextarea(parent: HTMLElement, label: string, value: string, onChange: (v: string) => void): void {
        const row = document.createElement('div');
        row.className = 'prop-row';
        row.style.flexDirection = 'column';
        row.style.alignItems = 'stretch';

        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;

        const ta = document.createElement('textarea');
        ta.className = 'prop-textarea';
        ta.value = value;
        ta.addEventListener('change', () => onChange(ta.value));

        row.appendChild(l);
        row.appendChild(ta);
        parent.appendChild(row);
    }

    _addCheckbox(parent: HTMLElement, label: string, value: boolean, onChange: (v: boolean) => void): void {
        const row = document.createElement('div');
        row.className = 'prop-row';

        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;

        const v = document.createElement('div');
        v.className = 'prop-value';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = Boolean(value);
        input.addEventListener('change', () => onChange(input.checked));

        v.appendChild(input);
        row.appendChild(l);
        row.appendChild(v);
        parent.appendChild(row);
    }

    _addSelect(parent: HTMLElement, label: string, value: string, options: string[], onChange: (v: string) => void): void {
        const row = document.createElement('div');
        row.className = 'prop-row';

        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;

        const v = document.createElement('div');
        v.className = 'prop-value';

        const select = document.createElement('select');
        select.className = 'prop-select';
        options.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt;
            o.text = opt;
            if (opt === value) o.selected = true;
            select.appendChild(o);
        });
        select.addEventListener('change', () => onChange(select.value));

        v.appendChild(select);
        row.appendChild(l);
        row.appendChild(v);
        parent.appendChild(row);
    }
}
