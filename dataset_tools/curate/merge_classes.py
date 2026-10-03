"""Fold sub-type classes into their parent label in datasets/waterfall + manifest.csv.

morse and remote-keyless-entry -> OOK (both are on/off keyed carriers);
8PSK, 16QAM, 32QAM -> psk-qam (indistinguishable on a waterfall).
* moves files to the parent folder and rewrites manifest rows (old label kept in params.subtype)
* synthetic keyfob captures made with the FSK variant are not OOK: archived to tmp/dataset/_merged_out/
* re-caps the merged legacy (real-old) images with the same farthest-point dHash selection
* writes dataset_tools/curate/known_frequencies.proposed.json with the merged frequency entry
Nothing is deleted; overflow goes to tmp/dataset/_merged_out/.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

from curate_existing import dhash, farthest_point
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, MANIFEST, ROOT, SCRATCH, TOOLS  # noqa: E402
REPO = TMP = ROOT  # manifest paths are relative to the repo root
MERGE = {"morse": "OOK", "remote-keyless-entry": "OOK",
         "8PSK": "psk-qam", "16QAM": "psk-qam", "32QAM": "psk-qam"}
LEGACY_CAP = 60
OUT = SCRATCH / "_merged_out"


def main():
    man = MANIFEST
    rows = list(csv.DictReader(open(man)))
    fields = list(rows[0].keys())
    kept, moved, archived = [], 0, 0
    for r in rows:
        if r["class"] in MERGE:
            params = json.loads(r["params"] or "{}")
            if r["source"] == "synthetic" and params.get("kind") == "2fsk":
                dst = OUT / r["file"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(TMP / r["file"], dst)
                archived += 1
                continue
            params.setdefault("subtype", r["class"])
            new_cls = MERGE[r["class"]]
            src = TMP / r["file"]
            dst = DATA / new_cls / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            r.update({"class": new_cls, "file": str(dst.relative_to(TMP)), "params": json.dumps(params),
                      "group_id": r["group_id"].replace(f"legacy-{params['subtype']}-", f"legacy-{new_cls}-")})
            moved += 1
        kept.append(r)
    # re-cap merged legacy images per parent class
    for parent in set(MERGE.values()):
        leg = [r for r in kept if r["class"] == parent and r["source"] == "real-old"]
        if len(leg) > LEGACY_CAP:
            sel = set(farthest_point([dhash(Image.open(TMP / r["file"])) for r in leg], LEGACY_CAP))
            drop = {id(r) for i, r in enumerate(leg) if i not in sel}
            for r in kept:
                if id(r) in drop:
                    dst = OUT / r["file"]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(TMP / r["file"], dst)
                    archived += 1
            kept = [r for r in kept if id(r) not in drop]
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(kept)
    for old in MERGE:
        d = DATA / old
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    kf = json.load(open(REPO / "known_frequencies.json"))
    merged = dict(kf)
    for old, new in MERGE.items():
        ranges = (merged.get(new) or []) + (merged.pop(old, None) or [])
        merged[new] = ranges or None
    (TOOLS / "curate" / "known_frequencies.proposed.json").write_text(json.dumps(merged, indent=4))
    print(f"relabelled {moved}, archived {archived}, rows now {len(kept)}")


if __name__ == "__main__":
    main()
