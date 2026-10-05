"""Cap every class at --target images without losing diversity.

* real-ota (live captures) are always kept - real-world data is the scarcest
* the remaining images are trimmed one at a time from the currently largest (subtype, capture group) bucket,
  so sub-types (e.g. BPSK/QPSK/8PSK/16QAM/32QAM in psk-qam) and capture sessions stay evenly represented
* overflow is moved (not deleted) to tmp/dataset/_balanced_out/

usage: python balance.py [--target 150] [--dry-run]
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import MANIFEST, ROOT, SCRATCH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    man = MANIFEST
    rows = list(csv.DictReader(open(man)))
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)
    drop = set()
    for cls, rs in by_cls.items():
        excess = len(rs) - a.target
        if excess <= 0:
            continue
        buckets = defaultdict(list)
        for r in rs:
            if r["source"] == "real-ota":
                continue
            sub = json.loads(r["params"] or "{}").get("subtype", r["source"])
            buckets[(sub, r["group_id"])].append(r)
        sub_count = Counter()
        for (sub, _), b in buckets.items():
            sub_count[sub] += len(b)
        for _ in range(excess):
            # largest sub-type first, then its largest capture group
            sub = max(sub_count, key=sub_count.get)
            key = max((k for k in buckets if k[0] == sub and buckets[k]), key=lambda k: len(buckets[k]))
            drop.add(id(buckets[key].pop()))
            sub_count[sub] -= 1
        print(f"{cls:32s} {len(rs)} -> {a.target}")
    keep = [r for r in rows if id(r) not in drop]
    if a.dry_run:
        return
    for r in rows:
        if id(r) in drop:
            dst = SCRATCH / "_balanced_out" / r["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(ROOT / r["file"], dst)
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(keep)
    c = Counter(r["class"] for r in keep)
    print(len(keep), "images;", sorted(c.items(), key=lambda x: x[1]))


if __name__ == "__main__":
    main()
