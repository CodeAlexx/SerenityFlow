/**
 * Serialize/deserialize workflow JSON.
 * Supports two formats:
 *   1. API format: { "id": { class_type, inputs: { name: value|[src,slot] } } }
 *   2. ComfyUI graph format: { nodes: [...], links: [...] }
 */

function serializeWorkflow(canvas) {
    const prompt = {};

    canvas.nodes.forEach((node, id) => {
        const inputs = {};

        node.inputs.forEach((input, idx) => {
            const conn = canvas.connections.find(
                c => c.targetNode === id && c.targetSlot === idx
            );
            if (conn) {
                inputs[input.name] = [conn.sourceNode, conn.sourceSlot];
            } else if (node.widgetValues[input.name] !== undefined) {
                inputs[input.name] = node.widgetValues[input.name];
            }
        });

        prompt[id] = {
            class_type: node.nodeType,
            inputs: inputs,
        };
    });

    return prompt;
}

/**
 * Detect format and load accordingly.
 */
function loadWorkflow(canvas, data, nodeInfo) {
    try {
        if (data.version && data.prompt) {
            console.log('[workflow] Loading native format');
            deserializeWorkflow(canvas, data.prompt, data.nodes, nodeInfo);
        } else if (data.nodes && Array.isArray(data.nodes)) {
            console.log('[workflow] Loading ComfyUI graph format,', data.nodes.length, 'top-level nodes');
            loadComfyUIGraph(canvas, data, nodeInfo);
        } else if (typeof data === 'object' && !Array.isArray(data)) {
            const firstVal = Object.values(data)[0];
            if (firstVal && firstVal.class_type) {
                console.log('[workflow] Loading raw API format');
                deserializeWorkflow(canvas, data, null, nodeInfo);
            } else {
                console.warn('[workflow] Unknown format, keys:', Object.keys(data).slice(0, 5));
            }
        }
    } catch (err) {
        console.error('[workflow] Load error:', err);
        throw err;
    }
}

/**
 * Load ComfyUI graph format (nodes[] + links[]).
 * Handles subgraph expansion.
 */
function loadComfyUIGraph(canvas, graphData, nodeInfo) {
    // Clear
    const existingIds = [...canvas.nodes.keys()];
    existingIds.forEach(id => canvas.removeNode(id));
    canvas.connections = [];

    // Collect subgraph definitions
    const subgraphDefs = {};
    if (graphData.definitions && graphData.definitions.subgraphs) {
        for (const sg of graphData.definitions.subgraphs) {
            subgraphDefs[sg.id] = sg;
        }
    }
    console.log('[workflow] Subgraph defs:', Object.keys(subgraphDefs).length);

    // Flatten: expand subgraphs into top-level nodes
    const allNodes = [];
    const allLinks = [];
    const skipTypes = new Set(['MarkdownNote', 'Reroute', 'Note']);

    for (const node of graphData.nodes) {
        const ntype = node.type || '';
        if (subgraphDefs[ntype]) {
            // Expand subgraph
            const sg = subgraphDefs[ntype];
            for (const inner of (sg.nodes || [])) {
                if (inner.id >= 0) allNodes.push(inner);
            }
            for (const link of (sg.links || [])) {
                allLinks.push(link);
            }
            console.log('[workflow] Expanded subgraph:', ntype.substring(0, 8) + '...');
        } else {
            allNodes.push(node);
        }
    }

    // Add outer links
    for (const link of (graphData.links || [])) {
        allLinks.push(link);
    }

    // Build link map: link_id -> { srcId, srcSlot, dstId, dstSlot, type }
    const linkMap = {};
    for (const link of allLinks) {
        let id, srcId, srcSlot, dstId, dstSlot, ltype;
        if (Array.isArray(link)) {
            [id, srcId, srcSlot, dstId, dstSlot, ltype] = link;
        } else {
            id = link.id;
            srcId = link.origin_id;
            srcSlot = link.origin_slot;
            dstId = link.target_id;
            dstSlot = link.target_slot;
            ltype = link.type || '*';
        }
        linkMap[id] = { srcId: String(srcId), srcSlot, dstId: String(dstId), dstSlot, type: ltype };
    }

    // Build a map of graph node inputs: graphNodeId -> [{name, link_id, slot}]
    const graphNodeInputs = {};
    for (const node of allNodes) {
        const nid = String(node.id);
        const inputs = node.inputs || [];
        graphNodeInputs[nid] = inputs.map((inp, idx) => ({
            name: inp.name || inp.label || '',
            link: inp.link,
            slot: idx,
        }));
    }

    // Create canvas nodes
    const idMap = {}; // graph node id -> canvas node id
    let created = 0;

    for (const node of allNodes) {
        const nid = String(node.id);
        const ntype = node.type || '';

        if (skipTypes.has(ntype) || !ntype || node.id < 0) continue;

        const info = nodeInfo ? nodeInfo[ntype] : null;

        // Position
        let px = 0, py = 0;
        const pos = node.pos;
        if (Array.isArray(pos) && pos.length >= 2) {
            px = pos[0]; py = pos[1];
        } else if (pos && typeof pos === 'object') {
            px = pos['0'] || 0; py = pos['1'] || 0;
        }

        try {
            const canvasNode = canvas.addNode(ntype, px, py, info);
            idMap[nid] = canvasNode.id;
            created++;

            // Apply widgets_values
            if (node.widgets_values && info) {
                _applyWidgetValues(canvasNode, node.widgets_values, info);
            }
        } catch (err) {
            console.error('[workflow] Failed to create node', nid, ntype, err);
        }
    }

    console.log('[workflow] Created', created, 'nodes');

    // Create connections using graph node input link references
    let connCount = 0;
    for (const node of allNodes) {
        const nid = String(node.id);
        const dstCanvasId = idMap[nid];
        if (!dstCanvasId) continue;

        const dstNode = canvas.nodes.get(dstCanvasId);
        if (!dstNode) continue;

        const inputs = node.inputs || [];
        for (let i = 0; i < inputs.length; i++) {
            const inp = inputs[i];
            const linkId = inp.link;
            if (linkId == null) continue;

            const link = linkMap[linkId];
            if (!link) continue;

            const srcCanvasId = idMap[link.srcId];
            if (!srcCanvasId) continue;

            // Find the target slot by name match in the canvas node
            const inputName = inp.name || inp.label || '';
            let targetSlot = dstNode.getInputIndex(inputName);

            // Fallback: use positional index if name doesn't match
            if (targetSlot < 0 && i < dstNode.inputs.length) {
                targetSlot = i;
            }

            if (targetSlot >= 0) {
                try {
                    canvas.addConnection(srcCanvasId, link.srcSlot, dstCanvasId, targetSlot);
                    connCount++;
                } catch (err) {
                    console.error('[workflow] Failed to connect', link.srcId, '->', nid, err);
                }
            }
        }
    }

    console.log('[workflow] Created', connCount, 'connections');

    canvas.nodeLayer.batchDraw();
    canvas.connectionLayer.batchDraw();
    canvas.centerView();

    // Update topbar model badge from the primary loader node
    _updateModelBadgeFromWorkflow(allNodes);
}

/**
 * Map widgets_values array to named input fields.
 * ComfyUI stores widget values as a flat array. Widget-type inputs appear
 * in the order defined by INPUT_TYPES (required then optional).
 */
function _applyWidgetValues(canvasNode, widgetValues, info) {
    if (!Array.isArray(widgetValues) || widgetValues.length === 0) return;

    // Get widget-type inputs in order
    const widgetInputs = [];
    for (const inp of canvasNode.inputs) {
        if (['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO'].includes(inp.type)) {
            widgetInputs.push(inp.name);
        }
    }

    let vi = 0;
    for (let i = 0; i < widgetInputs.length && vi < widgetValues.length; i++) {
        const name = widgetInputs[i];
        const value = widgetValues[vi];

        canvasNode.setWidgetValue(name, value);
        if (canvasNode.widgets[name]) {
            canvasNode.widgets[name].setValue(value);
        }
        vi++;

        // Some widgets consume an extra value (seed + control_after_generate)
        if ((name.includes('seed') || name === 'noise_seed') && vi < widgetValues.length) {
            const next = widgetValues[vi];
            if (typeof next === 'string' && ['fixed', 'increment', 'decrement', 'randomize'].includes(next)) {
                vi++;
            }
        }
    }
}

/**
 * Deserialize API format prompt.
 */
function deserializeWorkflow(canvas, prompt, nodePositions, nodeInfo) {
    const nodeIds = [...canvas.nodes.keys()];
    nodeIds.forEach(id => canvas.removeNode(id));
    canvas.connections = [];

    if (!prompt || typeof prompt !== 'object') return;

    let x = 100, y = 100;
    const idMap = {};

    for (const [origId, nodeData] of Object.entries(prompt)) {
        if (!nodeData || !nodeData.class_type) continue;

        const info = nodeInfo ? nodeInfo[nodeData.class_type] : null;

        let posX = x, posY = y;
        if (nodePositions && nodePositions[origId]) {
            const pos = nodePositions[origId].pos;
            if (pos) { posX = pos[0]; posY = pos[1]; }
        }

        const node = canvas.addNode(nodeData.class_type, posX, posY, info);
        idMap[origId] = node.id;

        for (const [name, value] of Object.entries(nodeData.inputs || {})) {
            if (!Array.isArray(value)) {
                node.setWidgetValue(name, value);
                if (node.widgets[name]) {
                    node.widgets[name].setValue(value);
                }
            }
        }

        const widgetVals = nodePositions && nodePositions[origId]
            ? nodePositions[origId].widgets_values
            : null;
        if (widgetVals) {
            for (const [name, value] of Object.entries(widgetVals)) {
                node.setWidgetValue(name, value);
                if (node.widgets[name]) {
                    node.widgets[name].setValue(value);
                }
            }
        }

        if (!nodePositions || !nodePositions[origId]) {
            y += 200;
            if (y > 1200) { y = 100; x += 300; }
        }
    }

    for (const [origId, nodeData] of Object.entries(prompt)) {
        if (!nodeData || !nodeData.inputs) continue;

        for (const [name, value] of Object.entries(nodeData.inputs)) {
            if (Array.isArray(value) && value.length === 2) {
                const sourceId = idMap[String(value[0])];
                const targetId = idMap[origId];
                const sourceSlot = value[1];

                if (!sourceId || !targetId) continue;

                const targetNode = canvas.nodes.get(targetId);
                if (!targetNode) continue;

                const targetSlot = targetNode.getInputIndex(name);
                if (targetSlot < 0) continue;

                canvas.addConnection(sourceId, sourceSlot, targetId, targetSlot);
            }
        }
    }

    canvas.centerView();

    // Update topbar model badge from the workflow
    _updateModelBadgeFromPrompt(prompt);
}

/**
 * Scan litegraph nodes for a model loader and update the topbar badge.
 */
function _updateModelBadgeFromWorkflow(nodes) {
    var loaderTypes = ['CheckpointLoaderSimple', 'UNETLoader', 'LTXVLoader'];
    for (var i = 0; i < nodes.length; i++) {
        var ntype = nodes[i].type || '';
        if (loaderTypes.indexOf(ntype) < 0) continue;
        var wvals = nodes[i].widgets_values;
        if (wvals && wvals.length > 0 && typeof wvals[0] === 'string') {
            _setModelBadge(wvals[0]);
            return;
        }
    }
}

/**
 * Scan API-format prompt for a model loader and update the topbar badge.
 */
function _updateModelBadgeFromPrompt(prompt) {
    var loaderTypes = ['CheckpointLoaderSimple', 'UNETLoader', 'LTXVLoader'];
    for (var id in prompt) {
        var nd = prompt[id];
        if (!nd || !nd.class_type) continue;
        if (loaderTypes.indexOf(nd.class_type) < 0) continue;
        var inputs = nd.inputs || {};
        var modelName = inputs.ckpt_name || inputs.unet_name || inputs.checkpoint_path;
        if (modelName && typeof modelName === 'string') {
            _setModelBadge(modelName);
            return;
        }
    }
}

/**
 * Set the topbar model badge text.
 */
function _setModelBadge(modelName) {
    var badge = document.querySelector('.model-badge');
    if (!badge) return;
    var short = modelName.split('/').pop().replace(/\.\w+$/, '');
    badge.textContent = short;
}
