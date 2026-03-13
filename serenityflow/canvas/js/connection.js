/**
 * Bezier connections between output and input ports.
 */
class SFConnection {
    constructor(sourceNodeId, sourceSlot, targetNodeId, targetSlot, canvas) {
        this.sourceNode = sourceNodeId;
        this.sourceSlot = sourceSlot;
        this.targetNode = targetNodeId;
        this.targetSlot = targetSlot;
        this.canvas = canvas;

        const srcType = this._getSourceType();

        this.line = new Konva.Shape({
            sceneFunc: (context, shape) => {
                const src = this._getSourcePos();
                const tgt = this._getTargetPos();
                const dx = Math.max(Math.abs(tgt.x - src.x) * 0.5, 30);

                context.beginPath();
                context.moveTo(src.x, src.y);
                context.bezierCurveTo(
                    src.x + dx, src.y,
                    tgt.x - dx, tgt.y,
                    tgt.x, tgt.y
                );
                context.fillStrokeShape(shape);
            },
            stroke: getTypeColor(srcType),
            strokeWidth: 2,
            hitStrokeWidth: 10,
            listening: true,
        });

        // Right-click to delete
        this.line.on('contextmenu', (e) => {
            e.evt.preventDefault();
            e.cancelBubble = true;
            // Will be handled by context menu system
        });

        this._watchDrag();
    }

    _getSourcePos() {
        const node = this.canvas.nodes.get(this.sourceNode);
        return node ? node.getOutputPortPos(this.sourceSlot) : { x: 0, y: 0 };
    }

    _getTargetPos() {
        const node = this.canvas.nodes.get(this.targetNode);
        return node ? node.getInputPortPos(this.targetSlot) : { x: 0, y: 0 };
    }

    _getSourceType() {
        const node = this.canvas.nodes.get(this.sourceNode);
        return node ? node.getOutputType(this.sourceSlot) : '*';
    }

    update() {
        const layer = this.line.getLayer();
        if (layer) layer.batchDraw();
    }

    _watchDrag() {
        const srcNode = this.canvas.nodes.get(this.sourceNode);
        const tgtNode = this.canvas.nodes.get(this.targetNode);
        if (srcNode) srcNode.group.on('dragmove.conn' + this.sourceNode + this.targetNode, () => this.update());
        if (tgtNode) tgtNode.group.on('dragmove.conn' + this.sourceNode + this.targetNode, () => this.update());
    }

    destroy() {
        // Clean up drag listeners
        const srcNode = this.canvas.nodes.get(this.sourceNode);
        const tgtNode = this.canvas.nodes.get(this.targetNode);
        const ns = '.conn' + this.sourceNode + this.targetNode;
        if (srcNode) srcNode.group.off('dragmove' + ns);
        if (tgtNode) tgtNode.group.off('dragmove' + ns);

        this.line.destroy();
    }
}
