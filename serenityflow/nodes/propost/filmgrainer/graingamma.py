# Filmgrainer - by Lars Ole Pontoppidan - MIT License
from __future__ import annotations

import numpy as np

_ShadowEnd = 160
_HighlightStart = 200


def _gammaCurve(gamma, x):
    """Returns from 0.0 to 1.0."""
    return pow((x / 255.0), (1.0 / gamma))


def _calcDevelopment(shadow_level, high_level, x):
    if x < _ShadowEnd:
        power = 0.5 - (_ShadowEnd - x) * (0.5 - shadow_level) / _ShadowEnd
    elif x < _HighlightStart:
        power = 0.5
    else:
        power = 0.5 - (x - _HighlightStart) * (0.5 - high_level) / (255 - _HighlightStart)
    return power


class Map:
    def __init__(self, map):
        self.map = map

    @staticmethod
    def calculate(src_gamma, noise_power, shadow_level, high_level) -> Map:
        map = np.zeros([256, 256], dtype=np.uint8)

        crop_top = noise_power * high_level / 12
        crop_low = noise_power * shadow_level / 20

        pic_scale = 1 - (crop_top + crop_low)
        pic_offs = 255 * crop_low

        for src_value in range(0, 256):
            pic_value = _gammaCurve(src_gamma, src_value) * 255.0

            gamma = pic_value * (1.5 / 256) + 0.5
            gamma_offset = _gammaCurve(gamma, 128)

            power = _calcDevelopment(shadow_level, high_level, pic_value)

            for noise_value in range(0, 256):
                gamma_compensated = _gammaCurve(gamma, noise_value) - gamma_offset
                value = pic_value * pic_scale + pic_offs + 255.0 * power * noise_power * gamma_compensated
                if value < 0:
                    value = 0
                elif value < 255.0:
                    value = int(value)
                else:
                    value = 255
                map[src_value, noise_value] = value

        return Map(map)

    def lookup(self, pic_value, noise_value):
        return self.map[pic_value, noise_value]
