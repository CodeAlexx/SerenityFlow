"""Numpy-only .cube LUT parser and applicator.

Reads Iridas .cube files (1D and 3D) and applies them via trilinear interpolation.
No external dependencies beyond numpy.
"""
from __future__ import annotations

import os

import numpy as np


class LUT3D:
    """A 3D lookup table loaded from a .cube file."""

    def __init__(self, table: np.ndarray, domain: np.ndarray, name: str = ""):
        self.table = table      # shape (size, size, size, 3), float32
        self.domain = domain    # shape (2, 3) — [[min_r, min_g, min_b], [max_r, max_g, max_b]]
        self.name = name
        self.size = table.shape[0]

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply this 3D LUT to an image array (H, W, 3) float32 in [0, 1]."""
        h, w, _ = image.shape
        flat = image.reshape(-1, 3).astype(np.float64)

        # Scale to LUT index space
        n = self.size - 1
        coords = flat * n
        coords = np.clip(coords, 0, n)

        # Integer and fractional parts
        lo = np.floor(coords).astype(np.int32)
        lo = np.clip(lo, 0, n - 1)
        hi = lo + 1
        hi = np.clip(hi, 0, n)
        frac = coords - lo

        fr, fg, fb = frac[:, 0], frac[:, 1], frac[:, 2]

        # Trilinear interpolation — 8 corners
        t = self.table
        c000 = t[lo[:, 0], lo[:, 1], lo[:, 2]]
        c001 = t[lo[:, 0], lo[:, 1], hi[:, 2]]
        c010 = t[lo[:, 0], hi[:, 1], lo[:, 2]]
        c011 = t[lo[:, 0], hi[:, 1], hi[:, 2]]
        c100 = t[hi[:, 0], lo[:, 1], lo[:, 2]]
        c101 = t[hi[:, 0], lo[:, 1], hi[:, 2]]
        c110 = t[hi[:, 0], hi[:, 1], lo[:, 2]]
        c111 = t[hi[:, 0], hi[:, 1], hi[:, 2]]

        # Interpolate along R
        fr3 = fr[:, np.newaxis]
        c00 = c000 * (1 - fr3) + c100 * fr3
        c01 = c001 * (1 - fr3) + c101 * fr3
        c10 = c010 * (1 - fr3) + c110 * fr3
        c11 = c011 * (1 - fr3) + c111 * fr3

        # Interpolate along G
        fg3 = fg[:, np.newaxis]
        c0 = c00 * (1 - fg3) + c10 * fg3
        c1 = c01 * (1 - fg3) + c11 * fg3

        # Interpolate along B
        fb3 = fb[:, np.newaxis]
        result = c0 * (1 - fb3) + c1 * fb3

        return result.reshape(h, w, 3).astype(np.float32)


def read_cube(path: str, clip: bool = False) -> LUT3D:
    """Parse an Iridas .cube file and return a LUT3D."""
    title = os.path.splitext(os.path.basename(path))[0]
    size = None
    domain_min = np.array([0.0, 0.0, 0.0])
    domain_max = np.array([1.0, 1.0, 1.0])
    values = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            keyword = parts[0].upper()

            if keyword == "TITLE":
                # Keep our filename-based title
                continue
            elif keyword == "LUT_3D_SIZE":
                size = int(parts[1])
            elif keyword == "DOMAIN_MIN":
                domain_min = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif keyword == "DOMAIN_MAX":
                domain_max = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif keyword == "LUT_1D_SIZE":
                raise ValueError("1D LUTs are not supported — only 3D .cube files")
            else:
                # Try to parse as RGB triplet
                try:
                    r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                    values.append([r, g, b])
                except (ValueError, IndexError):
                    continue

    if size is None:
        # Infer size from value count
        n = len(values)
        size = round(n ** (1.0 / 3.0))
        if size ** 3 != n:
            raise ValueError(f"Cannot infer LUT size: {n} values is not a perfect cube")

    expected = size ** 3
    if len(values) != expected:
        raise ValueError(f"Expected {expected} entries for size {size}, got {len(values)}")

    # .cube files store B as fastest-varying, then G, then R
    table = np.array(values, dtype=np.float32).reshape(size, size, size, 3)

    domain = np.array([domain_min, domain_max], dtype=np.float32)

    if clip:
        for dim in range(3):
            table[:, :, :, dim] = np.clip(table[:, :, :, dim], domain[0, dim], domain[1, dim])

    return LUT3D(table=table, domain=domain, name=title)
