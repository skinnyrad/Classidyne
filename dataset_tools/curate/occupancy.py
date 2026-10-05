"""Wideband occupancy check for live captures: fraction of frequency columns carrying energy clearly above the
noise floor. DC spikes / single spurs touch 1-2 columns, so empty-band frames score ~0 while real LTE / ATSC /
Wi-Fi frames occupy a large share of the span."""
import numpy as np
from PIL import Image


def occupancy(path, delta=12.0) -> float:
    a = np.asarray(Image.open(path).convert("L").resize((512, 256), Image.BOX), np.float32)
    col = a.mean(0)
    col = np.convolve(col, np.ones(5) / 5, mode="same")  # ignore 1-2 column spurs
    return float(np.mean(col > np.median(a) + delta))


if __name__ == "__main__":
    import csv
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import MANIFEST, ROOT
    cls = set(sys.argv[1:]) or {"cellular", "wifi"}
    for r in csv.DictReader(open(MANIFEST)):
        if r["source"] == "real-ota" and r["class"] in cls:
            print(f"{occupancy(ROOT / r['file']):.3f} {r['group_id']}")
