/**
 * Main canvas using Konva. Handles zoom, pan, background grid.
 */
class SFCanvas {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        this.stage = new Konva.Stage({
            container: containerId,
            width: this.container.offsetWidth,
            height: this.container.offsetHeight,
            draggable: true,
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

        this._setupZoom();
        this._setupResize();
        this._setupStageDrag();
        this._drawGrid();
    }

    _setupZoom() {
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
        });
    }

    _setupResize() {
        const ro = new ResizeObserver(() => {
            this.stage.width(this.container.offsetWidth);
            this.stage.height(this.container.offsetHeight);
            this._drawGrid();
        });
        ro.observe(this.container);
    }

    _setupStageDrag() {
        // Redraw grid on pan
        this.stage.on('dragmove', () => {
            this._drawGrid();
        });

        // Prevent stage drag when dragging a node
        this.stage.on('mousedown', (e) => {
            // Only allow stage drag if clicking on empty space or grid
            const clickedOnEmpty = e.target === this.stage || e.target.getLayer() === this.gridLayer;
            this.stage.draggable(clickedOnEmpty);
        });
    }

    _drawGrid() {
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

    addNode(nodeType, x, y, info) {
        const id = String(this.nextNodeId++);
        const node = new SFNode(id, nodeType, x, y, info, this);
        this.nodes.set(id, node);
        this.nodeLayer.add(node.group);
        this.nodeLayer.batchDraw();
        return node;
    }

    removeNode(id) {
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

    addConnection(sourceNodeId, sourceSlot, targetNodeId, targetSlot) {
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

    removeConnection(conn) {
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

    getConnectionsForNode(nodeId) {
        return this.connections.filter(c => c.sourceNode === nodeId || c.targetNode === nodeId);
    }

    isInputConnected(nodeId, slotIndex) {
        return this.connections.some(c => c.targetNode === nodeId && c.targetSlot === slotIndex);
    }

    getWorldPosition(screenX, screenY) {
        const transform = this.stage.getAbsoluteTransform().copy().invert();
        return transform.point({ x: screenX, y: screenY });
    }

    centerView() {
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

    selectNode(id, addToSelection) {
        if (!addToSelection) {
            this.deselectAll();
        }
        this.selectedNodes.add(id);
        const node = this.nodes.get(id);
        if (node) node.setSelected(true);
    }

    deselectAll() {
        this.selectedNodes.forEach(id => {
            const node = this.nodes.get(id);
            if (node) node.setSelected(false);
        });
        this.selectedNodes.clear();
    }

    deleteSelected() {
        const ids = [...this.selectedNodes];
        ids.forEach(id => this.removeNode(id));
    }

    // Start dragging a new connection from an output port
    startConnectionDrag(nodeId, slotIndex, startPos) {
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

    endConnectionDrag(targetNodeId, targetSlot) {
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

    cancelConnectionDrag() {
        this._cleanupConnectionDrag();
    }

    _cleanupConnectionDrag() {
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
}
