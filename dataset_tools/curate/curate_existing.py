"""Import the still-useful legacy waterfall images into datasets/waterfall (manifest: source=real-old).

Steps per class (the old `unknown` class is dropped):
  1. skip non-images / corrupt files / .DS_Store
  2. trim flat UI borders (near-constant rows/cols at the edges)
  3. quality filter: too small, nearly uniform, or extreme aspect ratio
  4. near-duplicate removal with a 16x16 difference hash (consecutive screenshots of one capture)
  5. group images into legacy "sessions" (same screenshot size + similar palette) and cap each session,
     so one capture session can no longer dominate a class
  6. pick a diverse subset (farthest-point on the hash) up to the per-class cap

usage: python curate_existing.py --src tmp/backups/datasets_v2/waterfall [--cap 100] [--cap-real-only 150] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, MANIFEST, ROOT, SCRATCH  # noqa: E402
REPO = TMP = ROOT  # manifest paths are relative to the repo root
OUT = DATA
REPORT = SCRATCH / "curate_report.json"
FIELDS = ["file", "class", "source", "group_id", "session", "rtl_center_hz", "offset_hz", "span_hz",
          "decimation", "fft_size", "fft_rate", "rtl_gain", "tx_gain", "min_db", "max_db", "tx_fs", "params"]

# Classes the RTL/HackRF bench cannot synthesize at full width: legacy + live HackRF captures only
REAL_ONLY = {"wifi", "atsc", "hdmi", "drone-video", "uav-video"}
DROP = {"unknown"}


def trim_borders(img: Image.Image, tol=4.0) -> Image.Image:
    a = np.asarray(img.convert("L"), np.float32)
    rows, cols = a.std(1), a.std(0)
    top = 0
    while top < len(rows) // 3 and rows[top] < tol:
        top += 1
    bot = len(rows)
    while bot > 2 * len(rows) // 3 and rows[bot - 1] < tol:
        bot -= 1
    left = 0
    while left < len(cols) // 3 and cols[left] < tol:
        left += 1
    right = len(cols)
    while right > 2 * len(cols) // 3 and cols[right - 1] < tol:
        right -= 1
    return img.crop((left, top, right, bot))


def dhash(img: Image.Image, n=16) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.LANCZOS), np.float32)
    return (g[:, 1:] > g[:, :-1]).ravel()


def palette_key(img: Image.Image) -> tuple:
    a = np.asarray(img.convert("RGB").resize((32, 32)), np.float32).reshape(-1, 3)
    return tuple((np.median(a, 0) // 48).astype(int))


def quality_ok(img: Image.Image) -> tuple[bool, str]:
    w, h = img.size
    if w < 200 or h < 150:
        return False, "too-small"
    if max(w / h, h / w) > 6:
        return False, "aspect"
    if np.asarray(img.convert("L"), np.float32).std() < 6:
        return False, "flat"
    return True, ""


def farthest_point(hashes: list[np.ndarray], k: int, seed=0) -> list[int]:
    if len(hashes) <= k:
        return list(range(len(hashes)))
    H = np.array(hashes, np.uint8)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(H)))]
    d = (H != H[chosen[0]]).sum(1)
    while len(chosen) < k:
        i = int(d.argmax())
        chosen.append(i)
        d = np.minimum(d, (H != H[i]).sum(1))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="legacy (v2) waterfall folder, e.g. tmp/backups/datasets_v2/waterfall")
    ap.add_argument("--cap", type=int, default=60, help="max legacy images per bench-synthesized class")
    ap.add_argument("--cap-real-only", type=int, default=150)
    ap.add_argument("--session-cap", type=float, default=0.4, help="max share of a class from one legacy session")
    ap.add_argument("--near-dup", type=int, default=12, help="dHash Hamming distance (of 256) = duplicate")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    report = {}
    rows_out = []
    for cdir in sorted(Path(a.src).iterdir()):
        if not cdir.is_dir() or cdir.name in DROP:
            continue
        cls = cdir.name
        stats = defaultdict(int)
        items = []
        for f in sorted(cdir.iterdir()):
            if f.name.startswith("."):
                stats["hidden"] += 1
                continue
            try:
                img = Image.open(f)
                img.load()
                img = img.convert("RGB")
            except Exception:
                stats["corrupt"] += 1
                continue
            img = trim_borders(img)
            ok, why = quality_ok(img)
            if not ok:
                stats[why] += 1
                continue
            items.append({"src": f, "img": img, "hash": dhash(img),
                          "session": f"{img.size[0] // 40}x{img.size[1] // 40}"})
        # near-duplicate removal (keep first of each cluster)
        kept = []
        for it in items:
            if any((it["hash"] != k["hash"]).sum() <= a.near_dup for k in kept):
                stats["near-dup"] += 1
                continue
            kept.append(it)
        cap = a.cap_real_only if cls in REAL_ONLY else a.cap
        # per-session cap
        by_sess = defaultdict(list)
        for it in kept:
            by_sess[it["session"]].append(it)
        sess_cap = max(5, int(a.session_cap * cap))
        pool = []
        for s, lst in by_sess.items():
            idx = farthest_point([x["hash"] for x in lst], min(len(lst), sess_cap))
            stats["session-capped"] += len(lst) - len(idx)
            pool += [lst[i] for i in idx]
        sel = [pool[i] for i in farthest_point([x["hash"] for x in pool], min(len(pool), cap))]
        stats["over-class-cap"] += len(pool) - len(sel)
        report[cls] = {"source": len(list(cdir.iterdir())), "kept": len(sel), "sessions": len(by_sess),
                       **dict(stats)}
        print(f"{cls:32s} {report[cls]}")
        if a.dry_run:
            continue
        d = OUT / cls
        d.mkdir(parents=True, exist_ok=True)
        for it in sel:
            tmpf = d / "_legacy_tmp.png"
            it["img"].save(tmpf, optimize=True)
            h = hashlib.sha256(tmpf.read_bytes()).hexdigest()
            dst = d / f"{h}.png"
            tmpf.rename(dst)
            rows_out.append({"file": str(dst.relative_to(TMP)), "class": cls, "source": "real-old",
                             "group_id": f"legacy-{cls}-{it['session']}", "session": f"legacy-{it['session']}",
                             "params": json.dumps({"legacy_file": it["src"].name, "size": list(it["img"].size)})})
    if not a.dry_run:
        new = not MANIFEST.exists()
        with MANIFEST.open("a", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows_out)
        REPORT.write_text(json.dumps(report, indent=2))
    print("total kept:", sum(r["kept"] for r in report.values()))


if __name__ == "__main__":
    main()
