/**
 * Shared type definitions for SerenityFlow canvas.
 * These are global (no import/export) — available to all script files.
 */

// ── ComfyUI /object_info response types ──

interface ComfyInputDef {
    required?: Record<string, ComfyInputSpec>;
    optional?: Record<string, ComfyInputSpec>;
    hidden?: Record<string, ComfyInputSpec>;
}

/** A single input spec: either [type, config] or [options_array, config] */
type ComfyInputSpec = [string | string[], Record<string, unknown>?];

interface ComfyNodeInfo {
    input?: ComfyInputDef;
    output?: string[];
    output_name?: string[];
    output_is_list?: boolean[];
    name?: string;
    display_name?: string;
    category?: string;
    description?: string;
    output_node?: boolean;
}

/** Full /object_info response: class_type -> node info */
type ComfyObjectInfo = Record<string, ComfyNodeInfo>;

// ── ComfyUI workflow/prompt types ──

interface ComfyPromptNode {
    class_type: string;
    inputs: Record<string, unknown>;
    _meta?: { title?: string };
}

/** A prompt dict: node_id -> node */
type ComfyPrompt = Record<string, ComfyPromptNode>;

// ── Gallery / generation metadata ──

interface GenerationMetadata {
    prompt?: string;
    model?: string | null;
    width?: number;
    height?: number;
    seed?: number | null;
    scheduler?: string | null;
    steps?: number | null;
    cfg?: number | null;
    guidance?: number | null;
    arch?: string;
    batchLabel?: string;
}

// ── Workflow builder params ──

interface WorkflowParams {
    model: string;
    prompt: string;
    negative?: string;
    negPrompt?: string;
    width: number;
    height: number;
    seed: number;
    steps: number;
    cfg: number;
    scheduler?: string;
    sampler?: string;
    denoise?: number;
    guidance?: number;
    batchSize?: number;
    clipSkip?: number;
    loras?: Array<{ name: string; strength: number; enabled?: boolean }>;
    arch?: string;
    frames?: number;
    fps?: number;
    imagePath?: string;
    maskPath?: string;
    initImageName?: string;
    maskImageName?: string;
    strengthModel?: number;
    strengthClip?: number;
    upscale?: string;
    [key: string]: unknown;
}

interface LoraParam {
    name: string;
    weight: number;
    clipWeight?: number;
}

// ── WebSocket event data (loosely typed — backend-defined shape) ──

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type WSEventData = any;

// ── Media output reference from ComfyUI ──

interface ComfyOutputFile {
    filename: string;
    subfolder: string;
    type: string;
}

// ── SFApi constructor-function interface ──

interface SFApi {
    connect(): void;
    on(type: string, fn: WSListener): void;
    off(type: string, fn: WSListener): void;
    interrupt(): Promise<Response>;
    viewUrl(filename: string, subfolder: string, type: string): string;
    getObjectInfo(): Promise<ComfyObjectInfo>;
    queuePrompt(workflow: unknown): Promise<unknown>;
    getClientId(): string;
}
