"""Evaluate the Classidyne app itself: the deployed model, the embeddings stored in classidyne_db and the
/api/classify voting rule (top-20 cosine neighbours, similarity >= 0.5, majority class).

1. leave-group-out  - every test-split image (its capture group was never trained on) is classified against the
                      real waterfall collection with its own capture group removed, so frames of the same
                      transmission / legacy session cannot vouch for each other
2. colormap shift   - the same test images re-rendered in every SDR++ colormap, embedded by the app's extractor
3. external images  - test_images/ and tests/lora.png (not part of the dataset)
4. HTTP             - a sample of queries sent to the running server: success, latency, agreement with (1)

usage (repo root, embedded DB; the server is only needed for part 4):
    python dataset_tools/eval/app_eval.py [--port 5000] [--http-n 40] [--out docs/APP_EVALUATION.md]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "gen")]
from paths import ROOT, SPLITS  # noqa: E402

K, THRESHOLD = 20, 0.5  # same as /api/classify
EXTERNAL = {"test_images/test1.png": "lora", "tests/lora.png": "lora", "test_images/test2.png": None}


def load_app():
    os.chdir(ROOT)  # app.py resolves model / DB / dataset paths relative to the repo root
    sys.path.insert(0, str(ROOT))
    import app
    return app


def vote(sims: np.ndarray, labels: np.ndarray):
    """/api/classify: top-K neighbours, keep similarity >= THRESHOLD, confidence = share of the kept votes."""
    top = np.argsort(-sims)[:K]
    kept = [labels[i] for i in top if sims[i] >= THRESHOLD]
    if not kept:
        return None, 0.0
    cls, n = Counter(kept).most_common(1)[0]
    return cls, 100.0 * n / len(kept)


def metrics(y, p):
    y, p = np.array(y, object), np.array(p, object)
    classes = sorted(set(y))
    f1, rec = [], {}
    for c in classes:
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        rec[c] = float(np.mean(p[y == c] == c))
    return float(np.mean(y == p)), float(np.mean(f1)), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLASSIDYNE_PORT", 5000)))
    ap.add_argument("--http-n", type=int, default=150)
    ap.add_argument("--out", default=str(ROOT / "docs" / "APP_EVALUATION.md"))
    a = ap.parse_args()
    app = load_app()

    # --- the app's stored embeddings (exactly what /api/classify searches)
    col = app.CLIENT.get_collection(app.WATERFALL_COLLECTION)
    db = col.get(include=["embeddings", "metadatas"])
    emb = np.asarray(db["embeddings"], np.float32)
    meta = db["metadatas"]
    by_path = {os.path.normpath(m["filepath"]): i for i, m in enumerate(meta)}
    labels = np.array([m["class"] for m in meta])
    splits = {os.path.normpath(r["file"]): r for r in csv.DictReader(open(SPLITS))}
    group = np.array([splits.get(os.path.normpath(m["filepath"]), {}).get("group_id", m["filehash"]) for m in meta])
    test = [(by_path[f], r) for f, r in splits.items() if r["split"] == "test" and f in by_path]
    missing = sum(1 for f in splits if f not in by_path)
    print(f"DB: {len(emb)} waterfall embeddings, {len(test)} test queries, {missing} manifest images not in DB")

    # 1. leave-group-out through the real DB
    y, p, conf, nomatch = [], [], [], 0
    by_src = defaultdict(lambda: ([], []))
    for i, r in test:
        sims = emb @ emb[i]
        sims[group == group[i]] = -1.0
        cls, c = vote(sims, labels)
        nomatch += cls is None
        y.append(r["class"])
        p.append(cls or "(no match)")
        conf.append(c)
        by_src[r["source"]][0].append(r["class"])
        by_src[r["source"]][1].append(cls or "(no match)")
    acc, f1, rec = metrics(y, p)
    conf, right = np.array(conf), np.array(y, object) == np.array(p, object)
    lines = ["# Classidyne app evaluation", "",
             f"Model: `{app.EXTRACTOR.arch}`, preprocessing: {'whole frame ' + 'x'.join(map(str, app.EXTRACTOR.input_size)) if app.EXTRACTOR.full_frame else 'resize + centre crop'}. "
             f"Waterfall collection: {len(emb)} images. Voting exactly as `/api/classify` "
             f"(top-{K}, similarity >= {THRESHOLD}).", "",
             "## 1. Held-out test images against the live vector DB (own capture group excluded)", "",
             "| queries | accuracy | macro-F1 | no match | mean confidence (right / wrong) |", "|---|---|---|---|---|",
             f"| {len(y)} | {acc:.3f} | {f1:.3f} | {nomatch} | {conf[right].mean():.0f}% / "
             f"{conf[~right].mean() if (~right).any() else float('nan'):.0f}% |", "",
             "| source | queries | accuracy | macro-F1 |", "|---|---|---|---|"]
    for src, (ys, ps) in sorted(by_src.items()):
        sa, sf, _ = metrics(ys, ps)
        lines.append(f"| {src} | {len(ys)} | {sa:.3f} | {sf:.3f} |")
    lines += ["", "Accuracy by reported confidence (how far the top-class percentage can be trusted):", "",
              "| confidence | share of queries | accuracy |", "|---|---|---|"]
    for lo, hi in ((0, 50), (50, 75), (75, 90), (90, 101)):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            lines.append(f"| {lo}-{min(hi, 100)}% | {m.mean():.0%} | {right[m].mean():.3f} |")
    conf_pairs = Counter((t, q) for t, q in zip(y, p) if t != q).most_common(10)
    lines += ["", "Most frequent confusions (true -> predicted):", ""]
    lines += [f"- {t} -> {q}: {n}" for (t, q), n in conf_pairs]
    lines += ["", "| class | test queries | recall |", "|---|---|---|"]
    for c in sorted(rec):
        lines.append(f"| {c} | {sum(t == c for t in y)} | {rec[c]:.2f} |")

    # 2. colormap shift: re-render SDR++ test images in every colormap, embed with the app's extractor
    from colormaps import gray_as, luts, to_level
    from common import colormap_of  # noqa: E402  (dataset_tools/train on the path below)
    cm_rows = [(i, r) for i, r in test if colormap_of(r)]
    lines += ["", f"## 2. Colormap robustness ({len(cm_rows)} SDR++ test captures re-rendered per colormap)", "",
              "| colormap | accuracy | macro-F1 |", "|---|---|---|"]
    levels = {}
    for i, r in cm_rows:
        img = Image.open(ROOT / r["file"]).convert("RGB")
        img.thumbnail((640, 640), Image.NEAREST)
        levels[i] = to_level(img, colormap_of(r))
    cm_scores = {}
    for name in luts():
        ys, ps = [], []
        for i, r in cm_rows:
            q = app.EXTRACTOR(gray_as(levels[i], name).convert("RGB"))
            sims = emb @ q.astype(np.float32)
            sims[group == group[i]] = -1.0
            ys.append(r["class"])
            ps.append(vote(sims, labels)[0] or "(no match)")
        ca, cf, _ = metrics(ys, ps)
        cm_scores[name] = cf
        lines.append(f"| {name} | {ca:.3f} | {cf:.3f} |")
        print(f"colormap {name}: acc {ca:.3f} F1 {cf:.3f}", flush=True)
    lines.append(f"| **mean** | | **{np.mean(list(cm_scores.values())):.3f}** |")

    # 3. external images
    lines += ["", "## 3. External images (not in the dataset)", "", "| image | expected | top class | confidence | runner-up |",
              "|---|---|---|---|---|"]
    for f, exp in EXTERNAL.items():
        if not (ROOT / f).exists():
            continue
        q = app.EXTRACTOR(Image.open(ROOT / f).convert("RGB")).astype(np.float32)
        sims = emb @ q
        top = np.argsort(-sims)[:K]
        kept = Counter(labels[j] for j in top if sims[j] >= THRESHOLD)
        tot = sum(kept.values()) or 1
        ranked = kept.most_common(2)
        first = f"{ranked[0][0]}" if ranked else "(no match)"
        lines.append(f"| `{f}` | {exp or '?'} | {first} | {100 * ranked[0][1] / tot if ranked else 0:.0f}% | "
                     f"{f'{ranked[1][0]} ({100 * ranked[1][1] / tot:.0f}%)' if len(ranked) > 1 else '-'} |")

    # 4. HTTP against the running server
    import httpx
    base = f"http://localhost:{a.port}"
    lines += ["", f"## 4. HTTP (`{base}`)", ""]
    try:
        stats = httpx.get(f"{base}/api/stats", timeout=10).json()
        rng = np.random.default_rng(0)
        pick = [test[j] for j in rng.choice(len(test), min(a.http_n, len(test)), replace=False)]
        lat, ok, top1 = [], 0, 0
        for i, r in pick:
            with open(ROOT / r["file"], "rb") as fh:
                t0 = time.time()
                res = httpx.post(f"{base}/api/classify", files={"query_image": ("q.png", fh, "image/png")},
                                 data={"collection": "waterfall"}, timeout=60).json()
                lat.append(time.time() - t0)
            ok += bool(res.get("success"))
            # the image itself is in the DB, so the live API answer includes its own capture group
            top1 += bool(res.get("class_scores")) and res["class_scores"][0]["class"] == r["class"]
        lines += [f"- `/api/stats`: `{stats}`",
                  f"- `/api/classify`: {ok}/{len(pick)} succeeded, median latency {np.median(lat) * 1000:.0f} ms "
                  f"(p90 {np.percentile(lat, 90) * 1000:.0f} ms)",
                  f"- top class correct for {top1}/{len(pick)} (includes the query's own capture group, so this is a "
                  f"sanity check, not an accuracy estimate)"]
    except httpx.HTTPError as e:
        lines.append(f"- server not reachable ({e}); start `python app.py` to run this part")

    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print("written", a.out)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
    main()
