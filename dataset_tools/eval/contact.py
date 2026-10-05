"""Contact sheet of captured images: python eval/contact.py <out.png> [class ...] [--n 6]"""
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import MANIFEST, ROOT  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("classes", nargs="*")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--cols", type=int, default=3)
a = ap.parse_args()
rows = [r for r in csv.DictReader(open(MANIFEST)) if not a.classes or r["class"] in a.classes]
picked = []
for c in dict.fromkeys(r["class"] for r in rows):
    picked += [r for r in rows if r["class"] == c][-a.n:]
W, H = 610, 264
sheet = Image.new("RGB", (a.cols * W, ((len(picked) + a.cols - 1) // a.cols) * (H + 16)), "black")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(picked):
    x, y = (i % a.cols) * W, (i // a.cols) * (H + 16)
    sheet.paste(Image.open(ROOT / r["file"]).resize((W - 4, H)), (x, y + 16))
    d.text((x + 3, y + 2), f"{r['class']} span={int(r['span_hz'])//1000}k rate={r['fft_rate']} {r['params'][:60]}", fill="white")
sheet.save(a.out)
print(len(picked), "tiles")
