"""Offline sanity check: render a falling-waterfall spectrogram for each generator (no RF).

usage: python preview.py [class ...]   -> tmp/dataset/preview/<class>.png + contact sheet
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from signals import GENERATORS

FS = 2_400_000
OUT = Path(__file__).resolve().parents[2] / "tmp" / "dataset" / "preview"
# Seconds of signal that fill one SDR++ screen for each class (slow modes need more)
DUR = {"morse": 8, "Radioteletype": 4, "sstv": 4, "automatic-picture-transmission": 4, "vor": 3,
       "packet": 4, "pocsag": 4, "remote-keyless-entry": 3, "RS41-Radiosonde": 4, "airband": 6, "am": 3}


def waterfall(iq, fs, rows=400, nfft=2048):
    hop = max(1, (len(iq) - nfft) // rows)
    win = np.hanning(nfft).astype(np.float32)
    spec = np.empty((rows, nfft), np.float32)
    for r in range(rows):
        seg = iq[r * hop: r * hop + nfft]
        if len(seg) < nfft:
            seg = np.pad(seg, (0, nfft - len(seg)))
        spec[r] = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(seg * win))) + 1e-6)
    lo, hi = np.percentile(spec, 5), spec.max()
    g = np.clip((spec - lo) / (hi - lo + 1e-9), 0, 1)
    return Image.fromarray((g * 255).astype(np.uint8)).resize((800, 400))


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    tiles = []
    for name in names:
        t0 = time.time()
        iq, meta = GENERATORS[name](rng, FS, DUR.get(name, 2))
        noise = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))) * 0.003
        img = waterfall(iq / (np.max(np.abs(iq)) + 1e-9) + noise, FS)
        img.save(OUT / f"{name}.png")
        tiles.append((name, img))
        print(f"{name:32s} {len(iq) / FS:5.1f}s  {time.time() - t0:5.1f}s  {meta}")
    cols = 4
    sheet = Image.new("L", (cols * 400, ((len(tiles) + cols - 1) // cols) * 220), 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(tiles):
        x, y = (i % cols) * 400, (i // cols) * 220
        sheet.paste(img.resize((396, 200)), (x, y + 18))
        d.text((x + 4, y + 2), name, fill=255)
    sheet.save(OUT / "_contact.png")


if __name__ == "__main__":
    main(sys.argv[1:] or list(GENERATORS))
