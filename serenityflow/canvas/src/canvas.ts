/**
 * Main canvas using Konva. Handles zoom, pan, background grid.
 */
class SFCanvas {
    // DOM & Konva
    container: HTMLElement;
    stage: Konva.Stage;
    gridLayer: Konva.Layer;
    connectionLayer: Konva.Layer;
    nodeLayer: Konva.Layer;
    overlayLayer: Konva.Layer;

    // State
    scale: number;
    nodes: Map<string, SFNode>;
    connections: SFConnection[];
    nextNodeId: number;
    nodeInfo: ComfyObjectInfo;
    selectedNodes: Set<string>;

    // Drag-connection state
    draggingConnection: { sourceNode: string; sourceSlot: number; startPos: { x: number; y: number }; endPos?: { x: number; y: number } } | null;
    tempLine: Konva.Shape | null;


    // Settings
    snapToGrid: boolean;
    gridSize: number;
    showMinimap: boolean;

    // Space-to-pan state
    _spaceDown: boolean;
    _prePanDraggable: boolean;

    // Cleanup registry
    _cleanups: Array<() => void>;

    // Connection drag handlers
    _connDragMoveHandler: ((e: Konva.KonvaEventObject<MouseEvent>) => void) | null = null;
    _connDragUpHandler: ((e: MouseEvent) => void) | null = null;

    // Minimap state
    _mmCanvas: HTMLCanvasElement | null = null;
    _mmCtx: CanvasRenderingContext2D | null = null;
    _mmSize: number = 0;
    _mmMoveHandler: ((e: MouseEvent) => void) | null = null;
    _mmUpHandler: (() => void) | null = null;

    constructor(containerId: string) {
        this.container = document.getElementById(containerId)!;

        this.stage = new Konva.Stage({
            container: containerId,
            width: this.container.offsetWidth,
            height: this.container.offsetHeight,
            draggable: false,
        });

        // Layers (order matters: grid behind connections behind nodes)
        this.gridLayer = new Konva.Layer({ listening: false });
        this.connectionLayer = new Konva.Layer();
        this.nodeLayer = new Konva.Layer();
        this.overlayLayer = new Konva.Layer(); // Selection box, drag preview

        this.stage.add(this.gridLayer);
        this.stage.add(this.connectionLayer);
        this.stage.add(this.nodeLayer);
        this.stage.add(this.overlayLayer);

        this.scale = 1;
        this.nodes = new Map();       // id -> SFNode
        this.connections = [];        // SFConnection[]
        this.nextNodeId = 1;
        this.nodeInfo = {};           // class_type -> object_info entry
        this.selectedNodes = new Set(); // Set of node IDs

        // Drag-connection state
        this.draggingConnection = null; // { sourceNode, sourceSlot, line }
        this.tempLine = null;

        // Settings
        this.snapToGrid = false;
        this.gridSize = 20;
        this.showMinimap = true;

        // Canvas container: suppress touch/selection defaults (InvokeAI pattern)
        this.container.style.touchAction = 'none';
        this.container.style.userSelect = 'none';
        this.container.style.webkitUserSelect = 'none';

        // Space-to-pan state
        this._spaceDown = false;
        this._prePanDraggable = false;

        // Cleanup registry for unmount
        this._cleanups = [];

        this._setupZoom();
        this._setupResize();
        this._setupStageDrag();
        this._setupMiddleMousePan();
        this._drawGrid();
        this._createViewportControls();
        this._createMinimap();
    }

    _setupZoom(): void {
        this.stage.on('wheel', (e) => {
            e.evt.preventDefault();

            const oldScale = this.stage.scaleX();
            const pointer = this.stage.getPointerPosition();
            if (!pointer) return;

            const scaleBy = 1.08;
            const direction = e.evt.deltaY > 0 ? -1 : 1;
            const newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;

            if (newScale < 0.1 || newScale > 5) return;

            this.scale = newScale;
            this.stage.scale({ x: newScale, y: newScale });

            const mousePointTo = {
                x: (pointer.x - this.stage.x()) / oldScale,
                y: (pointer.y - this.stage.y()) / oldScale,
            };
            const newPos = {
                x: pointer.x - mousePointTo.x * newScale,
                y: pointer.y - mousePointTo.y * newScale,
            };
            this.stage.position(newPos);
            this._drawGrid();
            this._updateMinimap();
        });
    }

    _setupResize(): void {
        const ro = new ResizeObserver(() => {
            this.stage.width(this.container.offsetWidth);
            this.stage.height(this.container.offsetHeight);
            this._drawGrid();
        });
        ro.observe(this.container);
    }

    // DOM-level panning. Stage stays draggable:false so it can't
    // conflict with Konva node drag.
    _isPanning: boolean = false;
    _panStartX: number = 0;
    _panStartY: number = 0;
    _stageStartX: number = 0;
    _stageStartY: number = 0;

    _setupStageDrag(): void {
        const container = this.stage.container();

        container.addEventListener('mousedown', (e: MouseEvent) => {
            // Middle mouse always pans
            if (e.button === 1) {
                this._startPan(e);
                e.preventDefault();
                return;
            }
            // Space+left-click pans
            if (e.button === 0 && this._spaceDown) {
                this._startPan(e);
                return;
            }
        });

        // Left-click on empty space pans (Konva event — fires after node handlers)
        this.stage.on('mousedown', (e) => {
            if (e.evt.button !== 0 || this._spaceDown) return;
            // Only pan if click hit the stage itself (empty space)
            if (e.target === this.stage) {
                this._startPan(e.evt);
            }
        });

        container.addEventListener('mousemove', (e: MouseEvent) => {
            if (!this._isPanning) return;
            this.stage.position({
                x: this._stageStartX + (e.clientX - this._panStartX),
                y: this._stageStartY + (e.clientY - this._panStartY),
            });
            this._drawGrid();
            this.stage.batchDraw();
        });

        container.addEventListener('mouseup', () => {
            if (this._isPanning) {
                this._isPanning = false;
                this.stage.container().style.cursor = '';
                this._updateMinimap();
            }
        });
    }

    _startPan(e: MouseEvent): void {
        this._isPanning = true;
        this._panStartX = e.clientX;
        this._panStartY = e.clientY;
        this._stageStartX = this.stage.x();
        this._stageStartY = this.stage.y();
        this.stage.container().style.cursor = 'grabbing';
    }

    _setupMiddleMousePan(): void {
        // consolidated into _setupStageDrag
    }

    // Space-to-pan: enable temporary pan mode
    startSpacePan(): void {
        if (this._spaceDown) return;
        this._spaceDown = true;
        this._prePanDraggable = false;
        this.stage.container().style.cursor = 'grab';
    }

    endSpacePan(): void {
        if (!this._spaceDown) return;
        this._spaceDown = false;
        this.stage.container().style.cursor = '';
    }

    _drawGrid(): void {
        this.gridLayer.destroyChildren();

        const gridSize = 20;
        const majorEvery = 5;
        const scale = this.stage.scaleX();
        const offset = this.stage.position();
        const width = this.stage.width();
        const height = this.stage.height();

        // Don't draw grid when too zoomed out
        if (scale < 0.3) {
            this.gridLayer.batchDraw();
            return;
        }

        const startX = Math.floor(-offset.x / scale / gridSize) * gridSize - gridSize;
        const startY = Math.floor(-offset.y / scale / gridSize) * gridSize - gridSize;
        const endX = startX + width / scale + gridSize * 2;
        const endY = startY + height / scale + gridSize * 2;

        const strokeWidth = 1 / scale;

        for (let x = startX; x < endX; x += gridSize) {
            const isMajor = Math.round(x / gridSize) % majorEvery === 0;
            this.gridLayer.add(new Konva.Line({
                points: [x, startY, x, endY],
                stroke: isMajor ? '#252545' : '#1e1e3e',
                strokeWidth: strokeWidth,
                listening: false,
            }));
        }

        for (let y = startY; y < endY; y += gridSize) {
            const isMajor = Math.round(y / gridSize) % majorEvery === 0;
            this.gridLayer.add(new Konva.Line({
                points: [startX, y, endX, y],
                stroke: isMajor ? '#252545' : '#1e1e3e',
                strokeWidth: strokeWidth,
                listening: false,
            }));
        }

        this.gridLayer.batchDraw();
    }

    addNode(nodeType: string, x: number, y: number, info: ComfyNodeInfo): SFNode {
        const id = String(this.nextNodeId++);
        const node = new SFNode(id, nodeType, x, y, info, this);
        this.nodes.set(id, node);
        this.nodeLayer.add(node.group);
        this.nodeLayer.batchDraw();
        return node;
    }

    removeNode(id: string): void {
        const node = this.nodes.get(id);
        if (!node) return;

        // Remove connections to/from this node
        this.connections = this.connections.filter(c => {
            if (c.sourceNode === id || c.targetNode === id) {
                c.destroy();
                return false;
            }
            return true;
        });

        node.destroy();
        this.nodes.delete(id);
        this.selectedNodes.delete(id);
        this.nodeLayer.batchDraw();
        this.connectionLayer.batchDraw();
    }

    addConnection(sourceNodeId: string, sourceSlot: number, targetNodeId: string, targetSlot: number): SFConnection {
        // Check if target input already has a connection — remove it
        this.connections = this.connections.filter(c => {
            if (c.targetNode === targetNodeId && c.targetSlot === targetSlot) {
                c.destroy();
                return false;
            }
            return true;
        });

        const conn = new SFConnection(sourceNodeId, sourceSlot, targetNodeId, targetSlot, this);
        this.connections.push(conn);
        this.connectionLayer.add(conn.line);
        this.connectionLayer.batchDraw();

        // Hide widget for connected input
        const targetNode = this.nodes.get(targetNodeId);
        if (targetNode) targetNode.updateWidgetVisibility();

        return conn;
    }

    removeConnection(conn: SFConnection): void {
        const idx = this.connections.indexOf(conn);
        if (idx >= 0) {
            this.connections.splice(idx, 1);
            conn.destroy();
            this.connectionLayer.batchDraw();

            // Show widget for disconnected input
            const targetNode = this.nodes.get(conn.targetNode);
            if (targetNode) targetNode.updateWidgetVisibility();
        }
    }

    getConnectionsForNode(nodeId: string): SFConnection[] {
        return this.connections.filter(c => c.sourceNode === nodeId || c.targetNode === nodeId);
    }

    isInputConnected(nodeId: string, slotIndex: number): boolean {
        return this.connections.some(c => c.targetNode === nodeId && c.targetSlot === slotIndex);
    }

    getWorldPosition(screenX: number, screenY: number): { x: number; y: number } {
        const transform = this.stage.getAbsoluteTransform().copy().invert();
        return transform.point({ x: screenX, y: screenY });
    }

    centerView(): void {
        if (this.nodes.size === 0) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + node.width);
            maxY = Math.max(maxY, node.y + node.height);
        });
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        const sw = this.stage.width();
        const sh = this.stage.height();
        this.stage.position({ x: sw / 2 - cx * this.scale, y: sh / 2 - cy * this.scale });
        this._drawGrid();
    }

    selectNode(id: string, addToSelection: boolean): void {
        if (!addToSelection) {
            this.deselectAll();
        }
        this.selectedNodes.add(id);
        const node = this.nodes.get(id);
        if (node) node.setSelected(true);
    }

    deselectAll(): void {
        this.selectedNodes.forEach(id => {
            const node = this.nodes.get(id);
            if (node) node.setSelected(false);
        });
        this.selectedNodes.clear();
    }

    deleteSelected(): void {
        const ids = [...this.selectedNodes];
        ids.forEach(id => this.removeNode(id));
    }

    // Start dragging a new connection from an output port
    startConnectionDrag(nodeId: string, slotIndex: number, startPos: { x: number; y: number }): void {
        this.draggingConnection = {
            sourceNode: nodeId,
            sourceSlot: slotIndex,
            startPos: startPos,
        };

        // Disable stage drag while connecting
        this.stage.draggable(false);

        const srcNode = this.nodes.get(nodeId);
        const srcType = srcNode ? srcNode.getOutputType(slotIndex) : '*';
        const wireColor = getTypeColor(srcType);

        this.tempLine = new Konva.Shape({
            sceneFunc: (context, shape) => {
                if (!this.draggingConnection) return;
                const src = this.draggingConnection.startPos;
                const end = this.draggingConnection.endPos || src;
                const dx = Math.max(Math.abs(end.x - src.x) * 0.5, 30);

                context.beginPath();
                context.moveTo(src.x, src.y);
                context.bezierCurveTo(
                    src.x + dx, src.y,
                    end.x - dx, end.y,
                    end.x, end.y
                );
                context.fillStrokeShape(shape);
            },
            stroke: wireColor,
            strokeWidth: 2,
            dash: [6, 3],
            listening: false,
        });

        this.overlayLayer.add(this.tempLine);

        // Track mouse movement to update the temp line
        this._connDragMoveHandler = (e) => {
            if (!this.draggingConnection) return;
            const pointer = this.stage.getPointerPosition();
            if (!pointer) return;
            this.draggingConnection.endPos = this.getWorldPosition(pointer.x, pointer.y);
            this.overlayLayer.batchDraw();
        };

        // End drag on mouseup anywhere on stage
        this._connDragUpHandler = (e) => {
            // If endConnectionDrag was already called by a port mouseup, this is a no-op
            if (this.draggingConnection) {
                this.cancelConnectionDrag();
            }
        };

        this.stage.on('mousemove.conndrag', this._connDragMoveHandler);

        // Use window mouseup with a tiny delay so port mouseup fires first
        window.addEventListener('mouseup', this._connDragUpHandler, { once: true });
    }

    endConnectionDrag(targetNodeId: string | null, targetSlot: number | undefined): void {
        if (!this.draggingConnection) return;

        if (targetNodeId != null && targetSlot !== undefined) {
            const srcNode = this.nodes.get(this.draggingConnection.sourceNode);
            const tgtNode = this.nodes.get(targetNodeId);
            if (srcNode && tgtNode) {
                const srcType = srcNode.getOutputType(this.draggingConnection.sourceSlot);
                const tgtType = tgtNode.getInputType(targetSlot);
                if (typesCompatible(srcType, tgtType)) {
                    if (this.draggingConnection.sourceNode !== targetNodeId) {
                        this.addConnection(
                            this.draggingConnection.sourceNode,
                            this.draggingConnection.sourceSlot,
                            targetNodeId,
                            targetSlot
                        );
                    }
                }
            }
        }

        this._cleanupConnectionDrag();
    }

    cancelConnectionDrag(): void {
        this._cleanupConnectionDrag();
    }

    _cleanupConnectionDrag(): void {
        if (this.tempLine) {
            this.tempLine.destroy();
            this.tempLine = null;
        }
        this.draggingConnection = null;
        this.stage.off('mousemove.conndrag');
        // Remove window listener if it hasn't fired yet
        if (this._connDragUpHandler) {
            window.removeEventListener('mouseup', this._connDragUpHandler);
            this._connDragUpHandler = null;
        }
        this._connDragMoveHandler = null;
        this.overlayLayer.batchDraw();
    }

    // ── Phase 1: Fit View ──
    fitView(animate?: boolean): void {
        if (this.nodes.size === 0) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + node.width);
            maxY = Math.max(maxY, node.y + node.height);
        });
        const pad = 60;
        const bw = maxX - minX + pad * 2;
        const bh = maxY - minY + pad * 2;
        const sw = this.stage.width();
        const sh = this.stage.height();
        const newScale = Math.min(sw / bw, sh / bh, 2);
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        const newX = sw / 2 - cx * newScale;
        const newY = sh / 2 - cy * newScale;

        if (animate) {
            const startScale = this.stage.scaleX();
            const startPos = this.stage.position();
            const duration = 300;
            const startTime = performance.now();
            const anim = () => {
                const t = Math.min((performance.now() - startTime) / duration, 1);
                const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // easeInOutQuad
                const s = startScale + (newScale - startScale) * ease;
                const x = startPos.x + (newX - startPos.x) * ease;
                const y = startPos.y + (newY - startPos.y) * ease;
                this.scale = s;
                this.stage.scale({ x: s, y: s });
                this.stage.position({ x, y });
                this._drawGrid();
                this._updateMinimap();
                if (t < 1) requestAnimationFrame(anim);
            };
            requestAnimationFrame(anim);
        } else {
            this.scale = newScale;
            this.stage.scale({ x: newScale, y: newScale });
            this.stage.position({ x: newX, y: newY });
            this._drawGrid();
            this._updateMinimap();
        }
    }

    // ── Phase 1: Snap to Grid ──
    snapNodeToGrid(node: SFNode): void {
        if (!this.snapToGrid) return;
        const g = this.gridSize;
        node.x = Math.round(node.x / g) * g;
        node.y = Math.round(node.y / g) * g;
        node.group.x(node.x);
        node.group.y(node.y);
        this.nodeLayer.batchDraw();
        this.connectionLayer.batchDraw();
    }

    // ── Phase 1: Viewport Controls ──
    _createViewportControls(): void {
        const html = `
            <div class="viewport-controls" id="viewport-controls">
                <button id="vc-zoom-in" title="Zoom In (+)">+</button>
                <button id="vc-zoom-out" title="Zoom Out (-)">−</button>
                <button id="vc-fit" title="Fit View (H)">⊞</button>
                <button id="vc-minimap" title="Toggle Minimap" class="active">◫</button>
                <button id="vc-snap" title="Snap to Grid (Shift+G)">#</button>
            </div>`;
        this.container.insertAdjacentHTML('beforeend', html);

        const self = this;
        document.getElementById('vc-zoom-in')!.addEventListener('click', () => {
            self.scale = Math.min(self.scale * 1.25, 5);
            self.stage.scale({ x: self.scale, y: self.scale });
            self._drawGrid();
            self._updateMinimap();
        });
        document.getElementById('vc-zoom-out')!.addEventListener('click', () => {
            self.scale = Math.max(self.scale / 1.25, 0.1);
            self.stage.scale({ x: self.scale, y: self.scale });
            self._drawGrid();
            self._updateMinimap();
        });
        document.getElementById('vc-fit')!.addEventListener('click', () => {
            self.fitView(true);
        });
        document.getElementById('vc-minimap')!.addEventListener('click', function(this: HTMLElement) {
            self.showMinimap = !self.showMinimap;
            this.classList.toggle('active', self.showMinimap);
            const mm = document.getElementById('canvas-minimap');
            if (mm) mm.style.display = self.showMinimap ? '' : 'none';
        });
        document.getElementById('vc-snap')!.addEventListener('click', function(this: HTMLElement) {
            self.snapToGrid = !self.snapToGrid;
            this.classList.toggle('active', self.snapToGrid);
        });
    }

    // ── Phase 1: Minimap ──
    _createMinimap(): void {
        const mmSize = 160;
        const html = `<canvas id="canvas-minimap" class="canvas-minimap" width="${mmSize}" height="${mmSize}"></canvas>`;
        this.container.insertAdjacentHTML('beforeend', html);
        this._mmCanvas = document.getElementById('canvas-minimap') as HTMLCanvasElement;
        this._mmCtx = this._mmCanvas.getContext('2d');
        this._mmSize = mmSize;

        // Click/drag on minimap to navigate
        const self = this;
        let mmDragging = false;
        const navigateFromMM = (e: MouseEvent) => {
            const rect = self._mmCanvas!.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const bounds = self._getNodeBounds();
            if (!bounds) return;
            const pad = 60;
            const bw = bounds.maxX - bounds.minX + pad * 2;
            const bh = bounds.maxY - bounds.minY + pad * 2;
            const mmScale = Math.min(mmSize / bw, mmSize / bh);
            const offX = (mmSize - bw * mmScale) / 2;
            const offY = (mmSize - bh * mmScale) / 2;
            const worldX = (mx - offX) / mmScale + bounds.minX - pad;
            const worldY = (my - offY) / mmScale + bounds.minY - pad;
            const sw = self.stage.width();
            const sh = self.stage.height();
            self.stage.position({
                x: sw / 2 - worldX * self.scale,
                y: sh / 2 - worldY * self.scale
            });
            self._drawGrid();
            self._updateMinimap();
        };
        this._mmCanvas.addEventListener('mousedown', (e) => { mmDragging = true; navigateFromMM(e); });
        this._mmMoveHandler = (e) => { if (mmDragging) navigateFromMM(e); };
        this._mmUpHandler = () => { mmDragging = false; };
        window.addEventListener('mousemove', this._mmMoveHandler);
        window.addEventListener('mouseup', this._mmUpHandler);

        // Update minimap on zoom/pan
        this.stage.on('dragmove.minimap', () => this._updateMinimap());
    }

    _getNodeBounds(): { minX: number; minY: number; maxX: number; maxY: number } | null {
        if (this.nodes.size === 0) return null;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + node.width);
            maxY = Math.max(maxY, node.y + node.height);
        });
        return { minX, minY, maxX, maxY };
    }

    _updateMinimap(): void {
        if (!this.showMinimap || !this._mmCtx) return;
        const ctx = this._mmCtx;
        const size = this._mmSize;
        ctx.clearRect(0, 0, size, size);

        const bounds = this._getNodeBounds();
        if (!bounds) return;

        const pad = 60;
        const bw = bounds.maxX - bounds.minX + pad * 2;
        const bh = bounds.maxY - bounds.minY + pad * 2;
        const mmScale = Math.min(size / bw, size / bh);
        const offX = (size - bw * mmScale) / 2;
        const offY = (size - bh * mmScale) / 2;

        // Draw nodes
        this.nodes.forEach(node => {
            const nx = (node.x - bounds.minX + pad) * mmScale + offX;
            const ny = (node.y - bounds.minY + pad) * mmScale + offY;
            const nw = node.width * mmScale;
            const nh = node.height * mmScale;
            const selected = this.selectedNodes.has(node.id);
            ctx.fillStyle = selected ? '#5b8def' : '#3a3a6a';
            ctx.fillRect(nx, ny, Math.max(nw, 3), Math.max(nh, 2));
        });

        // Draw viewport rect
        const stagePos = this.stage.position();
        const sw = this.stage.width();
        const sh = this.stage.height();
        const vpLeft = (-stagePos.x / this.scale - bounds.minX + pad) * mmScale + offX;
        const vpTop = (-stagePos.y / this.scale - bounds.minY + pad) * mmScale + offY;
        const vpW = (sw / this.scale) * mmScale;
        const vpH = (sh / this.scale) * mmScale;
        ctx.strokeStyle = 'rgba(91, 141, 239, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(vpLeft, vpTop, vpW, vpH);
        ctx.fillStyle = 'rgba(91, 141, 239, 0.08)';
        ctx.fillRect(vpLeft, vpTop, vpW, vpH);
    }

    // ── Phase 2: Auto-Layout ──
    autoLayout(): void {
        if (this.nodes.size === 0) return;

        // Build adjacency: node → [downstream nodes]
        const downstream = new Map();
        const upstream = new Map();
        this.nodes.forEach((_, id) => { downstream.set(id, new Set()); upstream.set(id, new Set()); });
        this.connections.forEach(c => {
            if (downstream.has(c.sourceNode)) downstream.get(c.sourceNode).add(c.targetNode);
            if (upstream.has(c.targetNode)) upstream.get(c.targetNode).add(c.sourceNode);
        });

        // Assign layers (longest path from sources, with cycle protection)
        const layers = new Map();
        const assignLayer = (id: string, depth: number, stack: Set<string>) => {
            if (stack.has(id)) return; // cycle detected — break
            const cur = layers.get(id) || 0;
            if (depth <= cur) return; // already visited at equal or greater depth
            layers.set(id, depth);
            stack.add(id);
            downstream.get(id).forEach((child: string) => assignLayer(child, depth + 1, new Set(stack)));
            stack.delete(id);
        };
        // Start from nodes with no upstream
        this.nodes.forEach((_, id) => {
            if (upstream.get(id).size === 0) assignLayer(id, 0, new Set());
        });
        // Handle disconnected nodes
        this.nodes.forEach((_, id) => { if (!layers.has(id)) layers.set(id, 0); });

        // Group nodes by layer
        const layerGroups = new Map();
        layers.forEach((layer, id) => {
            if (!layerGroups.has(layer)) layerGroups.set(layer, []);
            layerGroups.get(layer).push(id);
        });

        // Position nodes
        const hGap = 80;
        const vGap = 40;
        let xPos = 0;
        const sortedLayers = [...layerGroups.keys()].sort((a, b) => a - b);
        sortedLayers.forEach(layerIdx => {
            const nodeIds = layerGroups.get(layerIdx);
            let maxW = 0;
            let yPos = 0;
            nodeIds.forEach((id: string) => {
                const node = this.nodes.get(id);
                if (!node) return;
                node.x = xPos;
                node.y = yPos;
                node.group.x(xPos);
                node.group.y(yPos);
                maxW = Math.max(maxW, node.width);
                yPos += node.height + vGap;
            });
            xPos += maxW + hGap;
        });

        this.nodeLayer.batchDraw();
        this.connectionLayer.batchDraw();
        this.fitView(true);
        this._updateMinimap();
    }

    // ── Phase 1: Edge animation support ──
    setEdgesAnimated(animated: boolean): void {
        this.connections.forEach(conn => {
            if (conn.setAnimated) conn.setAnimated(animated);
        });
    }

    // Disconnect all connections for a given node
    disconnectAllForNode(nodeId: string): void {
        const toRemove = this.connections.filter(c => c.sourceNode === nodeId || c.targetNode === nodeId);
        toRemove.forEach(c => this.removeConnection(c));
    }

    // Cleanup all event listeners (call on unmount)
    destroy(): void {
        // Run registered cleanups
        this._cleanups.forEach(fn => fn());
        this._cleanups = [];

        // Remove minimap window listeners
        if (this._mmMoveHandler) window.removeEventListener('mousemove', this._mmMoveHandler);
        if (this._mmUpHandler) window.removeEventListener('mouseup', this._mmUpHandler);

        // Kill connection drag state
        this._cleanupConnectionDrag();

        // Destroy Konva stage
        this.stage.destroy();
    }
}
