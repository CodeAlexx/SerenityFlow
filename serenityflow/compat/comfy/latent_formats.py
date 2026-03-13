"""Compatibility shim for comfy.latent_formats.

Latent format specifications per model architecture.
"""
from __future__ import annotations


class LatentFormat:
    scale_factor = 1.0
    latent_channels = 4
    latent_rgb_factors = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor


class SD15(LatentFormat):
    scale_factor = 0.18215
    latent_channels = 4
    latent_rgb_factors = [
        [0.3512, 0.2297, 0.3227],
        [0.3250, 0.4974, 0.2350],
        [-0.2829, 0.1391, 0.2519],
        [-0.2120, -0.2616, -0.7177],
    ]


class SDXL(LatentFormat):
    scale_factor = 0.13025
    latent_channels = 4


class SDXL_Playground_2_5(LatentFormat):
    scale_factor = 0.5


class SD3(LatentFormat):
    scale_factor = 1.5305
    shift_factor = 0.0609
    latent_channels = 16

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor + self.shift_factor


class StableCascade_C(LatentFormat):
    latent_channels = 16
    scale_factor = 1.0


class StableCascade_B(LatentFormat):
    latent_channels = 4
    scale_factor = 1.0


class SC_Prior(StableCascade_C):
    pass


class SC_B(StableCascade_B):
    pass


class Flux(LatentFormat):
    scale_factor = 0.3611
    shift_factor = 0.1159
    latent_channels = 16

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor + self.shift_factor


class LTXV(LatentFormat):
    scale_factor = 1.0
    latent_channels = 128


class HunyuanVideo(LatentFormat):
    scale_factor = 0.476986
    latent_channels = 16
