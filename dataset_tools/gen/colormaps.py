"""SDR++ colormap round-trip augmentation.

Grayscale does not neutralise colormaps: most SDR++ maps are not monotonic in luminance (Classic goes
dark-blue -> white -> yellow -> red -> dark red, so the strongest signals turn *dark* in grayscale).
For images rendered in Classic we map every pixel back to its colormap index (= signal level), and at
training time re-colour that level image with a random SDR++ colormap before the usual grayscale step.

SDR++ interpolates the colour stops of each map linearly, so a 256-entry LUT reproduces it.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

CMAP_DIR = Path(os.environ.get("SDRPP_COLORMAPS", "/Applications/SDR++.app/Contents/Resources/colormaps"))


def _lut(stops: list[str]) -> np.ndarray:
    rgb = np.array([[int(s[i:i + 2], 16) for i in (1, 3, 5)] for s in stops], np.float32)
    x = np.linspace(0, 1, len(rgb))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, rgb[:, c]) for c in range(3)], 1).round().astype(np.uint8)


@lru_cache(None)
def luts() -> dict[str, np.ndarray]:
    out = {}
    for p in sorted(CMAP_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        out[d["name"]] = _lut(d["map"])
    if "Classic" not in out:
        raise FileNotFoundError(f"SDR++ colormaps not found in {CMAP_DIR} (set SDRPP_COLORMAPS)")
    return out


@lru_cache(None)
def _inverse(name="Classic", q=4) -> np.ndarray:
    """(256/q)^3 table: quantised RGB -> nearest LUT index (fast inverse colormap)."""
    lut = luts()[name].astype(np.float32)
    g = (np.arange(0, 256, q) + q / 2).astype(np.float32)
    cube = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    idx = np.empty(len(cube), np.uint8)
    for i in range(0, len(cube), 65536):
        d = ((cube[i:i + 65536, None, :] - lut[None]) ** 2).sum(-1)
        idx[i:i + 65536] = d.argmin(1)
    n = 256 // q
    return idx.reshape(n, n, n)


def to_level(img: Image.Image, name="Classic", q=4) -> Image.Image:
    """RGB waterfall rendered with colormap ``name`` -> 'L' image of colormap indices (signal level)."""
    a = np.asarray(img.convert("RGB")) // q
    return Image.fromarray(_inverse(name, q)[a[..., 0], a[..., 1], a[..., 2]])


def recolor(level: Image.Image, name: str) -> Image.Image:
    return Image.fromarray(luts()[name][np.asarray(level)])


def gray_as(level: Image.Image, name: str) -> Image.Image:
    """What app.py would see (convert('L')) for this waterfall rendered in colormap ``name``."""
    return recolor(level, name).convert("L")
