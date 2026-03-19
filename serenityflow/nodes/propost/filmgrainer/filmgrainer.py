# Filmgrainer - by Lars Ole Pontoppidan - MIT License
from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image, ImageFilter

from serenityflow.nodes.propost.filmgrainer import graingamma, graingen


def _grainTypes(typ):
    if typ == 1:
        return (0.8, 63)   # more interesting fine grain
    elif typ == 2:
        return (1, 45)     # basic fine grain
    elif typ == 3:
        return (1.5, 50)   # coarse grain
    elif typ == 4:
        return (1.6666, 50)  # coarser grain
    else:
        raise ValueError("Unknown grain type: " + str(typ))


MASK_CACHE_PATH = os.path.join(tempfile.gettempdir(), "mask-cache")


def _getGrainMask(img_width, img_height, saturation, grayscale, grain_size, grain_gauss, seed):
    if grayscale:
        str_sat = "BW"
        sat = -1.0
    else:
        str_sat = str(saturation)
        sat = saturation

    filename = os.path.join(
        MASK_CACHE_PATH,
        "grain-%d-%d-%s-%s-%s-%d.png" % (img_width, img_height, str_sat, str(grain_size), str(grain_gauss), seed),
    )
    if os.path.isfile(filename):
        mask = Image.open(filename)
    else:
        mask = graingen.grainGen(img_width, img_height, grain_size, grain_gauss, sat, seed)
        os.makedirs(MASK_CACHE_PATH, exist_ok=True)
        mask.save(filename, format="png", compress_level=1)
    return mask


def process(image, scale, src_gamma, grain_power, shadows, highs,
            grain_type, grain_sat, gray_scale, sharpen, seed):
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image).convert("RGB")
    org_width = img.size[0]
    org_height = img.size[1]

    if scale != 1.0:
        img = img.resize(
            (int(org_width / scale), int(org_height / scale)),
            resample=Image.LANCZOS,
        )

    img_width = img.size[0]
    img_height = img.size[1]

    map = graingamma.Map.calculate(src_gamma, grain_power, shadows, highs)

    (grain_size, grain_gauss) = _grainTypes(grain_type)
    mask = _getGrainMask(img_width, img_height, grain_sat, gray_scale, grain_size, grain_gauss, seed)

    mask_pixels = mask.load()
    img_pixels = img.load()
    lookup = map.map

    if gray_scale:
        for y in range(img_height):
            for x in range(img_width):
                m = mask_pixels[x, y]
                (r, g, b) = img_pixels[x, y]
                gray = int(0.21 * r + 0.72 * g + 0.07 * b)
                gray_lookup = lookup[gray, m]
                img_pixels[x, y] = (gray_lookup, gray_lookup, gray_lookup)
    else:
        for y in range(img_height):
            for x in range(img_width):
                (mr, mg, mb) = mask_pixels[x, y]
                (r, g, b) = img_pixels[x, y]
                r = lookup[r, mr]
                g = lookup[g, mg]
                b = lookup[b, mb]
                img_pixels[x, y] = (r, g, b)

    if scale != 1.0:
        img = img.resize((org_width, org_height), resample=Image.LANCZOS)

    if sharpen > 0:
        for _ in range(sharpen):
            img = img.filter(ImageFilter.SHARPEN)

    return np.array(img).astype("float32") / 255.0
