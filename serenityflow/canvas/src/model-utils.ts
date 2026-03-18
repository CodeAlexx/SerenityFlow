/**
 * Model Utilities — SerenityFlow
 * Architecture detection from model filenames and dimension clamping.
 */

interface ResolutionPreset {
    label: string;
    width: number;
    height: number;
}

interface ModelUtilEntry {
    name: string;
    loader: 'checkpoint' | 'unet';
}

type ArchType = 'sd15' | 'flux' | 'sd3' | 'sdxl' | 'ltxv' | 'wan' | 'klein' | 'qwen' | 'zimage' | 'hunyuan';

type ObjectInfo = ComfyObjectInfo | null;

interface ModelUtilsAPI {
    detectArchFromFilename(filename: string | null | undefined): ArchType;
    isVideoModel(filename: string | null | undefined): boolean;
    IMAGE_RESOLUTIONS: ResolutionPreset[];
    VIDEO_RESOLUTIONS: ResolutionPreset[];
    snapTo64(val: number): number;
    snapTo32(val: number): number;
    clampDimension(val: number): number;
    clampVideoDimension(val: number): number;
    fetchAllModels(): Promise<ModelUtilEntry[]>;
    loadObjectInfo(): Promise<ObjectInfo>;
    clearCache(): void;
}

var ModelUtils: ModelUtilsAPI = (function(): ModelUtilsAPI {
    'use strict';

    var objectInfoCache: ObjectInfo = null;

    function loadObjectInfo(): Promise<ObjectInfo> {
        if (objectInfoCache) return Promise.resolve(objectInfoCache);

        var cachedEtag: string | null = localStorage.getItem('sf-object-info-etag');
        var headers: Record<string, string> = {};
        if (cachedEtag) headers['If-None-Match'] = cachedEtag;

        return fetch('/object_info', { headers: headers })
            .then(function(resp: Response): Promise<ObjectInfo> | ObjectInfo {
                if (resp.status === 304) {
                    try {
                        var cached: ObjectInfo = JSON.parse(localStorage.getItem('sf-object-info-data')!);
                        if (cached) { objectInfoCache = cached; return cached; }
                    } catch(e) {}
                }
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.json().then(function(data: ObjectInfo): ObjectInfo {
                    objectInfoCache = data;
                    var etag: string | null = resp.headers.get('ETag');
                    if (etag) {
                        localStorage.setItem('sf-object-info-etag', etag);
                        try {
                            localStorage.setItem('sf-object-info-data', JSON.stringify(data));
                        } catch(e) {
                            localStorage.removeItem('sf-object-info-data');
                        }
                    }
                    return data;
                });
            });
    }

    // Detect architecture from model filename heuristics.
    // The backend does proper header detection — this is frontend-only best-effort
    // for choosing the right workflow graph before generation starts.
    function detectArchFromFilename(filename: string | null | undefined): ArchType {
        if (!filename) return 'sd15';
        var f: string = filename.toLowerCase();
        if (f.includes('qwen')) return 'qwen';
        if (f.includes('zimage') || f.includes('z-image') || f.includes('z_image')) return 'zimage';
        if (f.includes('flux') || f.includes('flex') || f.includes('f1d') || f.includes('f1s')) return 'flux';
        if (f.includes('sd3') || f.includes('stable-diffusion-3') || f.includes('sd_3')) return 'sd3';
        if (f.includes('sdxl') || f.includes('xl') || f.includes('pony') || f.includes('illustrious')) return 'sdxl';
        if (f.includes('ltx') || f.includes('ltxv')) return 'ltxv';
        if (f.includes('wan')) return 'wan';
        if (f.includes('klein')) return 'klein';
        if (f.includes('hunyuan') || f.includes('capybara')) return 'hunyuan';
        return 'sd15';
    }

    // Check if a detected architecture is a video model
    function isVideoModel(filename: string | null | undefined): boolean {
        var arch: ArchType = detectArchFromFilename(filename);
        return arch === 'ltxv' || arch === 'wan';
    }

    // Standard image resolutions (1024-based, snap to 64)
    var IMAGE_RESOLUTIONS: ResolutionPreset[] = [
        { label: '1:1',  width: 1024, height: 1024 },
        { label: '4:3',  width: 1152, height: 896  },
        { label: '16:9', width: 1344, height: 768  },
        { label: '3:4',  width: 896,  height: 1152 },
        { label: '9:16', width: 768,  height: 1344 }
    ];

    // Standard video resolutions (smaller, snap to 32 for video VAE)
    var VIDEO_RESOLUTIONS: ResolutionPreset[] = [
        { label: '1:1',  width: 512, height: 512 },
        { label: '4:3',  width: 768, height: 576 },
        { label: '16:9', width: 768, height: 432 },
        { label: '3:4',  width: 576, height: 768 },
        { label: '9:16', width: 432, height: 768 }
    ];

    // Snap a dimension value to nearest multiple of 64
    function snapTo64(val: number): number {
        return Math.max(64, Math.round(val / 64) * 64);
    }

    // Snap a dimension value to nearest multiple of 32 (video VAE requirement)
    function snapTo32(val: number): number {
        return Math.max(64, Math.round(val / 32) * 32);
    }

    // Validate and clamp a dimension: min 256, max 4096, divisible by 64
    function clampDimension(val: number): number {
        return Math.min(4096, Math.max(256, snapTo64(val)));
    }

    // Validate and clamp a video dimension: min 64, max 1280, divisible by 32
    function clampVideoDimension(val: number): number {
        return Math.min(1280, Math.max(64, snapTo32(val)));
    }

    // Fetch all available models from /object_info, merging checkpoints and UNETs.
    // Returns a promise that resolves to an array of { name, loader } objects.
    // loader is 'checkpoint' or 'unet'.
    // Filter out sub-model components (text encoders, clips, sharded parts, upscalers, loras)
    function isMainModel(name: string): boolean {
        var lower: string = name.toLowerCase();
        // Skip files inside subdirectories that are clearly sub-components
        if (/\/(text_encoder|clip|tokenizer|vae|scheduler|feature_extractor|vision_encoder|transformer)\//.test(name)) return false;
        // Skip sharded model parts (model-00001-of-00004.safetensors)
        if (/model-\d+-of-\d+/.test(lower)) return false;
        // Skip upscalers and loras mixed in
        if (/upscaler|upscale/.test(lower)) return false;
        if (/[\-_]lora[\-_\.]/.test(lower)) return false;
        // Skip individual training checkpoints (transformer-0001, etc.)
        if (/transformer-\d{4}/.test(lower)) return false;
        // Skip diffusers subdirectory components (e.g. capybara/transformer/model.safetensors)
        if (/\/transformer\//.test(name)) return false;
        return true;
    }

    function fetchAllModels(): Promise<ModelUtilEntry[]> {
        return loadObjectInfo()
            .then(function(data: ObjectInfo): ModelUtilEntry[] {
                var models: ModelUtilEntry[] = [];
                var seen: Record<string, boolean> = {};

                // Checkpoints (SD1.5, SDXL, SD3, etc.)
                var ckptInfo = data && data.CheckpointLoaderSimple &&
                    data.CheckpointLoaderSimple.input &&
                    data.CheckpointLoaderSimple.input.required &&
                    data.CheckpointLoaderSimple.input.required.ckpt_name;
                if (ckptInfo && Array.isArray(ckptInfo[0])) {
                    ckptInfo[0].forEach(function(m: string): void {
                        if (!seen[m] && isMainModel(m)) {
                            seen[m] = true;
                            models.push({ name: m, loader: 'checkpoint' });
                        }
                    });
                }

                // UNETs (FLUX, Klein, WAN, etc.)
                var unetInfo = data && data.UNETLoader &&
                    data.UNETLoader.input &&
                    data.UNETLoader.input.required &&
                    data.UNETLoader.input.required.unet_name;
                if (unetInfo && Array.isArray(unetInfo[0])) {
                    unetInfo[0].forEach(function(m: string): void {
                        if (!seen[m] && isMainModel(m)) {
                            seen[m] = true;
                            models.push({ name: m, loader: 'unet' });
                        }
                    });
                }

                return models;
            });
    }

    function clearCache(): void {
        objectInfoCache = null;
        localStorage.removeItem('sf-object-info-etag');
        localStorage.removeItem('sf-object-info-data');
    }

    return {
        detectArchFromFilename: detectArchFromFilename,
        isVideoModel: isVideoModel,
        IMAGE_RESOLUTIONS: IMAGE_RESOLUTIONS,
        VIDEO_RESOLUTIONS: VIDEO_RESOLUTIONS,
        snapTo64: snapTo64,
        snapTo32: snapTo32,
        clampDimension: clampDimension,
        clampVideoDimension: clampVideoDimension,
        fetchAllModels: fetchAllModels,
        loadObjectInfo: loadObjectInfo,
        clearCache: clearCache
    };
})();
