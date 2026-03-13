/**
 * Entry point. Bootstraps canvas, sidebar, toolbar, etc.
 */

// Global references
let sfCanvas, sfApi, sfSidebar, sfToolbar, sfProperties, sfPreview, sfHistory, sfSelection;

document.addEventListener('DOMContentLoaded', async () => {
    // Create API client
    sfApi = new SFApi();

    // Create canvas
    sfCanvas = new SFCanvas('canvas-container');

    // Create subsystems
    sfSidebar = new SFSidebar(sfCanvas, sfApi);
    sfProperties = new SFProperties(sfCanvas);
    sfPreview = new SFPreview(sfApi);
    sfHistory = new SFHistory(sfCanvas);
    sfSelection = new SFSelection(sfCanvas);
    sfToolbar = new SFToolbar(sfCanvas, sfApi);

    // Setup context menu
    setupContextMenu(sfCanvas, sfSidebar);

    // Setup drag-drop from sidebar to canvas
    setupSidebarDragDrop(sfCanvas, sfSidebar);

    // Setup double-click on node to open properties
    sfCanvas.nodeLayer.on('dblclick', (e) => {
        // Find which node was clicked
        let target = e.target;
        while (target && target.getParent() !== sfCanvas.nodeLayer) {
            target = target.getParent();
        }
        if (target) {
            const nodeId = findNodeIdByGroup(sfCanvas, target);
            if (nodeId) sfProperties.show(nodeId);
        }
    });

    // Connect to server
    sfApi.connect();

    // Load node types
    await sfSidebar.loadNodeTypes();
});

function findNodeIdByGroup(canvas, group) {
    for (const [id, node] of canvas.nodes) {
        if (node.group === group) return id;
    }
    return null;
}

// --- Context Menu ---

function setupContextMenu(canvas, sidebar) {
    const menu = document.getElementById('context-menu');
    const menuItems = document.getElementById('context-menu-items');
    let contextTarget = null; // { type: 'canvas'|'node'|'connection', data }

    // Hide on click anywhere
    document.addEventListener('click', () => {
        menu.classList.add('hidden');
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            menu.classList.add('hidden');
        }
    });

    // Right-click on canvas stage
    canvas.stage.on('contextmenu', (e) => {
        e.evt.preventDefault();

        const target = e.target;
        const pointer = canvas.stage.getPointerPosition();
        if (!pointer) return;

        menuItems.innerHTML = '';

        // Check what was right-clicked
        let clickedNode = null;
        let clickedConnection = null;

        // Walk up to find node group
        let t = target;
        while (t && t.getParent() !== canvas.nodeLayer) {
            t = t.getParent();
        }
        if (t) {
            clickedNode = findNodeIdByGroup(canvas, t);
        }

        // Check connections
        if (!clickedNode) {
            for (const conn of canvas.connections) {
                if (target === conn.line) {
                    clickedConnection = conn;
                    break;
                }
            }
        }

        if (clickedNode) {
            // Node context menu
            addMenuItem(menuItems, 'Properties', () => sfProperties.show(clickedNode));
            addMenuItem(menuItems, 'Duplicate', () => {
                const node = canvas.nodes.get(clickedNode);
                if (node) {
                    const newNode = canvas.addNode(node.nodeType, node.x + 30, node.y + 30, node.info);
                    for (const [name, val] of Object.entries(node.widgetValues)) {
                        newNode.setWidgetValue(name, val);
                        if (newNode.widgets[name]) newNode.widgets[name].setValue(val);
                    }
                }
            });
            addSeparator(menuItems);
            addMenuItem(menuItems, 'Delete', () => {
                sfHistory.saveState();
                canvas.removeNode(clickedNode);
            });
        } else if (clickedConnection) {
            // Connection context menu
            addMenuItem(menuItems, 'Delete Connection', () => {
                sfHistory.saveState();
                canvas.removeConnection(clickedConnection);
            });
        } else {
            // Canvas context menu - add nodes
            addMenuItem(menuItems, 'Add Node...', null, true);
            addSeparator(menuItems);

            // Quick-add for common categories
            const categories = Object.keys(sidebar.categories).sort().slice(0, 15);
            for (const cat of categories) {
                const nodes = sidebar.categories[cat];
                if (nodes.length <= 8) {
                    // Inline category
                    const catItem = document.createElement('div');
                    catItem.className = 'context-submenu-title';
                    catItem.textContent = cat;
                    menuItems.appendChild(catItem);

                    for (const nodeName of nodes) {
                        const info = sidebar.nodeTypes[nodeName];
                        addMenuItem(menuItems, '  ' + (info.display_name || nodeName), () => {
                            const worldPos = canvas.getWorldPosition(pointer.x, pointer.y);
                            canvas.addNode(nodeName, worldPos.x, worldPos.y, info);
                        });
                    }
                } else {
                    addMenuItem(menuItems, cat + ' (' + nodes.length + ')', () => {
                        // TODO: submenu
                    });
                }
            }
        }

        // Position menu
        menu.style.left = e.evt.clientX + 'px';
        menu.style.top = e.evt.clientY + 'px';
        menu.classList.remove('hidden');

        // Keep menu on screen
        const rect = menu.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            menu.style.left = (window.innerWidth - rect.width - 4) + 'px';
        }
        if (rect.bottom > window.innerHeight) {
            menu.style.top = (window.innerHeight - rect.height - 4) + 'px';
        }
    });
}

function addMenuItem(container, label, onClick, disabled) {
    const item = document.createElement('div');
    item.className = 'context-item';
    item.textContent = label;
    if (disabled) {
        item.style.color = 'var(--text-muted)';
        item.style.cursor = 'default';
    } else if (onClick) {
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            document.getElementById('context-menu').classList.add('hidden');
            onClick();
        });
    }
    container.appendChild(item);
}

function addSeparator(container) {
    const sep = document.createElement('div');
    sep.className = 'context-separator';
    container.appendChild(sep);
}

// --- Drag and Drop from sidebar ---

function setupSidebarDragDrop(canvas, sidebar) {
    const container = document.getElementById('canvas-container');

    container.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });

    container.addEventListener('drop', (e) => {
        e.preventDefault();
        const nodeType = e.dataTransfer.getData('text/plain');
        if (!nodeType) return;

        const rect = container.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;

        // Convert to stage coordinates
        const stageBox = canvas.stage.container().getBoundingClientRect();
        const pointerX = e.clientX - stageBox.left;
        const pointerY = e.clientY - stageBox.top;

        sidebar.addNodeAtPosition(nodeType, pointerX, pointerY);
    });
}
