/**
 * Workflow Builder — SerenityFlow
 * Model-aware ComfyUI workflow graph construction for all supported architectures.
 * Shared by Generate tab and Canvas tab.
 */

var WorkflowBuilder = (function() {
    'use strict';

    function build(params) {
        var arch = ModelUtils.detectArchFromFilename(params.model);
        var workflow;
        switch (arch) {
            case 'flux':  workflow = buildFlux(params); break;
            case 'sd3':   workflow = buildSD3(params); break;
            case 'sdxl':  workflow = buildSDXL(params); break;
            case 'ltxv':  workflow = buildLTXV(params); break;
            case 'wan':   workflow = buildWan(params); break;
            default:      workflow = buildSD15(params); break;
        }
        if (params.loras && params.loras.length > 0) {
            workflow = injectLoRAs(workflow, params.loras);
        }
        return workflow;
    }

    function buildImg2Img(params) {
        var arch = ModelUtils.detectArchFromFilename(params.model);
        var workflow;
        switch (arch) {
            case 'flux':  workflow = buildFluxImg2Img(params); break;
            case 'sd3':   workflow = buildSD3Img2Img(params); break;
            case 'sdxl':  workflow = buildSDXLImg2Img(params); break;
            default:      workflow = buildSD15Img2Img(params); break;
        }
        if (params.loras && params.loras.length > 0) {
            workflow = injectLoRAs(workflow, params.loras);
        }
        return workflow;
    }

    function buildInpaint(params) {
        // For now, use img2img workflow when mask is present.
        // Full inpaint nodes (SetLatentNoiseMask) can be added later.
        var workflow = buildImg2Img(params);
        if (!workflow) return build(params);
        return workflow;
    }

    function resolveSeed(seed) {
        return seed === -1 ? Math.floor(Math.random() * 4294967296) : seed;
    }

    /**
     * Inject LoRA loader nodes into a completed workflow.
     * Finds the model/clip source nodes, chains LoRA loaders, and rewires
     * downstream consumers to use the last LoRA output.
     */
    function injectLoRAs(workflow, loras) {
        // Find model and clip source nodes
        var modelNodeId = null;
        var clipNodeId = null;
        var modelIsCheckpoint = false;

        Object.keys(workflow).forEach(function(key) {
            var node = workflow[key];
            if (node.class_type === 'CheckpointLoaderSimple') {
                modelNodeId = key;
                clipNodeId = key;
                modelIsCheckpoint = true;
            } else if (node.class_type === 'UNETLoader' || node.class_type === 'LTXVLoader') {
                modelNodeId = key;
            } else if (node.class_type === 'DualCLIPLoader' || node.class_type === 'CLIPLoader') {
                clipNodeId = key;
            }
        });

        if (!modelNodeId) return workflow;

        var nextId = Math.max.apply(null, Object.keys(workflow).map(Number)) + 1;
        var origModelOut = [modelNodeId, 0];
        var origClipOut = clipNodeId ? [clipNodeId, modelIsCheckpoint ? 1 : 0] : null;
        var prevModelRef = origModelOut;
        var prevClipRef = origClipOut;

        loras.forEach(function(lora) {
            var id = String(nextId++);
            var inputs = {
                lora_name: lora.name,
                strength_model: lora.strength,
                strength_clip: lora.strength,
                model: prevModelRef
            };
            if (prevClipRef) {
                inputs.clip = prevClipRef;
            }
            workflow[id] = { class_type: 'LoraLoader', inputs: inputs };
            prevModelRef = [id, 0];
            if (prevClipRef) prevClipRef = [id, 1];
        });

        // Rewire nodes that referenced the original model/clip outputs
        Object.keys(workflow).forEach(function(key) {
            var node = workflow[key];
            if (node.class_type === 'LoraLoader') return;
            if (!node.inputs) return;
            Object.keys(node.inputs).forEach(function(k) {
                var v = node.inputs[k];
                if (!Array.isArray(v) || v.length < 2) return;
                // Rewire model references
                if (v[0] === origModelOut[0] && v[1] === origModelOut[1] &&
                    (k === 'model' || k === 'ltxv_model')) {
                    node.inputs[k] = prevModelRef;
                }
                // Rewire clip references (only direct clip, not conditioning)
                if (origClipOut && v[0] === origClipOut[0] && v[1] === origClipOut[1] && k === 'clip') {
                    node.inputs[k] = prevClipRef;
                }
            });
        });

        return workflow;
    }

    // ─── FLUX ───────────────────────────────────────────────────────────────

    function buildFlux(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var guidance = p.guidance || 3.5;
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'DualCLIPLoader', inputs: {
                clip_name1: 'clip_l.safetensors',
                clip_name2: 't5xxl_fp16.safetensors',
                type: 'flux'
            }},
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['2', 0] } },
            '5': { class_type: 'EmptySD3LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '6': { class_type: 'FluxGuidance', inputs: { conditioning: ['4', 0], guidance: guidance } },
            '7': { class_type: 'KSampler', inputs: {
                seed: seed, steps: p.steps, cfg: 1.0,
                sampler_name: 'euler', scheduler: 'simple', denoise: 1.0,
                model: ['1', 0], positive: ['6', 0], negative: ['4', 0], latent_image: ['5', 0]
            }},
            '8': { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['3', 0] } },
            '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'sf_generate' } }
        };
    }

    function buildFluxImg2Img(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var guidance = p.guidance || 3.5;
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'DualCLIPLoader', inputs: {
                clip_name1: 'clip_l.safetensors',
                clip_name2: 't5xxl_fp16.safetensors',
                type: 'flux'
            }},
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['2', 0] } },
            '5': { class_type: 'LoadImage', inputs: { image: p.initImageName } },
            '6': { class_type: 'VAEEncode', inputs: { pixels: ['5', 0], vae: ['3', 0] } },
            '7': { class_type: 'FluxGuidance', inputs: { conditioning: ['4', 0], guidance: guidance } },
            '8': { class_type: 'KSampler', inputs: {
                seed: seed, steps: p.steps, cfg: 1.0,
                sampler_name: 'euler', scheduler: 'simple', denoise: p.denoise || 0.75,
                model: ['1', 0], positive: ['7', 0], negative: ['4', 0], latent_image: ['6', 0]
            }},
            '9': { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['3', 0] } },
            '10': { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'sf_canvas' } }
        };
    }

    // ─── SD3 ────────────────────────────────────────────────────────────────

    function buildSD3(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'EmptySD3LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '5': { class_type: 'KSamplerAdvanced', inputs: {
                add_noise: 'enable', noise_seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                sampler_name: 'euler', scheduler: 'sgm_uniform', start_at_step: 0, end_at_step: 10000,
                return_with_leftover_noise: 'disable', denoise: 1.0,
                model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0]
            }},
            '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
            '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: 'sf_generate' } }
        };
    }

    function buildSD3Img2Img(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'LoadImage', inputs: { image: p.initImageName } },
            '5': { class_type: 'VAEEncode', inputs: { pixels: ['4', 0], vae: ['1', 2] } },
            '6': { class_type: 'KSamplerAdvanced', inputs: {
                add_noise: 'enable', noise_seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                sampler_name: 'euler', scheduler: 'sgm_uniform', start_at_step: 0, end_at_step: 10000,
                return_with_leftover_noise: 'disable', denoise: p.denoise || 0.75,
                model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['5', 0]
            }},
            '7': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['1', 2] } },
            '8': { class_type: 'SaveImage', inputs: { images: ['7', 0], filename_prefix: 'sf_canvas' } }
        };
    }

    // ─── SDXL ───────────────────────────────────────────────────────────────

    function buildSDXL(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'EmptyLatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '5': { class_type: 'KSampler', inputs: {
                seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                sampler_name: p.scheduler || 'euler', scheduler: 'normal', denoise: 1.0,
                model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0]
            }},
            '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
            '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: 'sf_generate' } }
        };
    }

    function buildSDXLImg2Img(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'LoadImage', inputs: { image: p.initImageName } },
            '5': { class_type: 'VAEEncode', inputs: { pixels: ['4', 0], vae: ['1', 2] } },
            '6': { class_type: 'KSampler', inputs: {
                seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                sampler_name: p.scheduler || 'euler', scheduler: 'normal', denoise: p.denoise || 0.75,
                model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['5', 0]
            }},
            '7': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['1', 2] } },
            '8': { class_type: 'SaveImage', inputs: { images: ['7', 0], filename_prefix: 'sf_canvas' } }
        };
    }

    // ─── SD1.5 ──────────────────────────────────────────────────────────────

    function buildSD15(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'EmptyLatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '5': { class_type: 'KSampler', inputs: {
                seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                sampler_name: p.scheduler || 'euler', scheduler: 'normal', denoise: 1.0,
                model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0]
            }},
            '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
            '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: 'sf_generate' } }
        };
    }

    function buildSD15Img2Img(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'LoadImage', inputs: { image: p.initImageName } },
            '5': { class_type: 'VAEEncode', inputs: { pixels: ['4', 0], vae: ['1', 2] } },
            '6': { class_type: 'KSampler', inputs: {
                seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                sampler_name: p.scheduler || 'euler', scheduler: 'normal', denoise: p.denoise || 0.75,
                model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['5', 0]
            }},
            '7': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['1', 2] } },
            '8': { class_type: 'SaveImage', inputs: { images: ['7', 0], filename_prefix: 'sf_canvas' } }
        };
    }

    // ─── LTX-V (Video) ──────────────────────────────────────────────────

    function buildLTXV(p) {
        var w = ModelUtils.clampVideoDimension(p.width);
        var h = ModelUtils.clampVideoDimension(p.height);
        var seed = resolveSeed(p.seed);
        var frames = Math.max(9, p.frames || 97);
        var fps = p.fps || 24;
        return {
            '1': { class_type: 'LTXVLoader', inputs: {
                checkpoint_path: p.model,
                gemma_path: 'gemma-3-12b-it',
                dtype: 'bfloat16'
            }},
            '2': { class_type: 'LTXVSampler', inputs: {
                ltxv_model: ['1', 0],
                prompt: p.prompt,
                negative_prompt: p.negPrompt || 'worst quality, inconsistent motion, blurry, jittery, distorted',
                width: w,
                height: h,
                num_frames: frames,
                steps: p.steps || 40,
                cfg: 3.0,
                seed: seed,
                frame_rate: fps,
                stg_scale: 1.0,
                mode: 'auto'
            }},
            '3': { class_type: 'SaveVideo', inputs: {
                video: ['2', 0],
                filename_prefix: 'sf_video',
                fps: fps,
                format: 'mp4'
            }}
        };
    }

    // ─── WAN (Video) ──────────────────────────────────────────────────────

    function buildWan(p) {
        var w = ModelUtils.clampVideoDimension(p.width);
        var h = ModelUtils.clampVideoDimension(p.height);
        var seed = resolveSeed(p.seed);
        var frames = Math.max(9, p.frames || 81);
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'umt5-xxl-enc-bf16.safetensors', type: 'wan' } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['2', 0] } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['2', 0] } },
            '5': { class_type: 'EmptyLatentVideo', inputs: { width: w, height: h, length: frames, batch_size: 1 } },
            '6': { class_type: 'KSampler', inputs: {
                seed: seed, steps: p.steps || 30, cfg: p.cfg || 5.0,
                sampler_name: 'euler', scheduler: 'normal', denoise: 1.0,
                model: ['1', 0], positive: ['3', 0], negative: ['4', 0], latent_image: ['5', 0]
            }},
            '7': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['1', 1] } },
            '8': { class_type: 'SaveAnimatedWEBP', inputs: {
                images: ['7', 0],
                filename_prefix: 'sf_video',
                fps: p.fps || 16
            }}
        };
    }

    return { build: build, buildImg2Img: buildImg2Img, buildInpaint: buildInpaint };
})();
