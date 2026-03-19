# Filmgrainer - by Lars Ole Pontoppidan - MIT License
from __future__ import annotations

import random

import numpy as np
from PIL import Image


def _makeGrayNoise(width, height, power):
    buffer = np.zeros([height, width], dtype=int)
    for y in range(height):
        for x in range(width):
            buffer[y, x] = random.gauss(128, power)
    buffer = buffer.clip(0, 255)
    return Image.fromarray(buffer.astype(dtype=np.uint8))


def _makeRgbNoise(width, height, power, saturation):
    buffer = np.zeros([height, width, 3], dtype=int)
    intens_power = power * (1.0 - saturation)
    for y in range(height):
        for x in range(width):
            intens = random.gauss(128, intens_power)
            buffer[y, x, 0] = random.gauss(0, power) * saturation + intens
            buffer[y, x, 1] = random.gauss(0, power) * saturation + intens
            buffer[y, x, 2] = random.gauss(0, power) * saturation + intens
    buffer = buffer.clip(0, 255)
    return Image.fromarray(buffer.astype(dtype=np.uint8))


def grainGen(width, height, grain_size, power, saturation, seed=1):
    noise_width = int(width / grain_size)
    noise_height = int(height / grain_size)
    random.seed(seed)

    if saturation < 0.0:
        img = _makeGrayNoise(noise_width, noise_height, power)
    else:
        img = _makeRgbNoise(noise_width, noise_height, power, saturation)

    if grain_size != 1.0:
        img = img.resize((width, height), resample=Image.LANCZOS)

    return img
