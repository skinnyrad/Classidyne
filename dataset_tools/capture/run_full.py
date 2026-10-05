"""Resumable, balanced full bench run.

Round-robin: each round runs one capture session (3 variants x 2 frames) for every class that is still
below its target, so stopping at any time leaves the classes balanced. Progress comes from manifest.csv,
so re-running simply continues.

usage: python run_full.py [--target 150] [--hours 1.9] [--only cls ...]
       python run_full.py --colormap random --hires --target 20   # top up 20 hi-res non-Classic images per class
"""
import argparse
import subprocess
import csv
import sys
import time
from collections import Counter

import run_capture as rc
from signals import CLASS_OF, GENERATORS

LIVE_PLANNED = {"cellular": 50, "bluetooth": 90}  # filled by run_live.py (HackRF RX)
LIVE_PLANNED_CMAP = {}  # colormap top-up: share captured live (run_live.py --colormap)


def counts(colormap_only=False):
    """Images per class; colormap_only counts only captures in a colormap other than Classic."""
    if not rc.MANIFEST.exists():
        return Counter()
    return Counter(r["class"] for r in csv.DictReader(open(rc.MANIFEST))
                   if not colormap_only or r.get("colormap") not in (None, "", "Classic"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--hours", type=float, default=1.9)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--colormap", help='SDR++ colormap or "random"; targets then count non-Classic captures only')
    ap.add_argument("--hires", action="store_true", help="every session uses a big FFT + fast waterfall")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--frames", type=int, default=2)
    a = ap.parse_args()
    cm_only = bool(a.colormap and a.colormap != "Classic")
    live = LIVE_PLANNED_CMAP if cm_only else LIVE_PLANNED
    deadline = time.time() + a.hours * 3600
    classes = list(dict.fromkeys(CLASS_OF.get(g, g) for g in GENERATORS))
    if a.only:
        classes = [c for c in classes if c in a.only]
    rnd = int(time.time()) % 100000
    while time.time() < deadline:
        c = counts(cm_only)
        todo = [k for k in classes if c[k] < a.target - live.get(k, 0)]
        if not todo:
            print("all classes at target", flush=True)
            return
        todo.sort(key=lambda k: c[k])  # emptiest first
        for k in todo:
            if time.time() > deadline:
                break
            t0 = time.time()
            # declare user activity: resets the idle timer so the screensaver / screen lock never starts
            subprocess.run(["caffeinate", "-u", "-t", "2"])
            cmap = a.colormap
            if cmap == "random":  # rotate per round (offset per class) so every class gets distinct colormaps
                others = [n for n in rc.colormaps.luts() if n != "Classic"]
                cmap = others[(rnd + classes.index(k)) % len(others)]
            rc.run_class(k, sessions=1, variants=a.variants, frames=a.frames, seed=rnd, colormap=cmap,
                         hires=True if a.hires else None)
            print(f"== {k}: {counts(cm_only)[k]} images, session took {time.time() - t0:.0f}s", flush=True)
        rnd += 1
    print("time budget reached", flush=True)


if __name__ == "__main__":
    import atexit
    import signal
    atexit.register(rc._stop_all)
    signal.signal(signal.SIGTERM, rc._on_signal)
    signal.signal(signal.SIGINT, rc._on_signal)
    main()
