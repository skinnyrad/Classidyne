"""Compare RadioNet (current) vs RadioNet v3 on the held-out test split.

For each model x preprocessing it reports, on images whose capture group was never trained on:
  * Classidyne kNN: top-20 cosine vote against the train+val gallery (what /api/classify does)
  * classifier head accuracy (v3 only)
  * accuracy, macro-F1, per-class recall, confusion matrix (PNG)
  * cross-domain: gallery = synthetic only -> queries = real images (and the reverse)
  * style-leak baseline: kNN on image size/aspect/mean colour only
  * --cmap-shift: SDR++-rendered test images re-rendered in every SDR++ colormap (gallery unchanged)

usage: python evaluate.py [--v3 models/RadioNet_v3_with_head.pth] [--out report.md]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import (MODELS, REPORTS, TMP, RadioNetV3, build_backbone, cached_gray, cached_level, colormap_of, device, load_splits,
                    tensorize)

HERE = REPORTS


@torch.no_grad()
def embed(model_fn, rows, mode, dev, bs=64, load=None, size=(224, 224)):
    load = load or (lambda r: cached_gray(r["file"]))
    feats, logits = [], []
    for i in range(0, len(rows), bs):
        x = torch.stack([tensorize(load(r), mode, size) for r in rows[i:i + bs]]).to(dev)
        f, lg = model_fn(x)
        feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
        if lg is not None:
            logits.append(lg.cpu())
    return torch.cat(feats).numpy(), (torch.cat(logits).numpy() if logits else None)


def knn(gal_f, gal_y, q_f, k=20):
    s = q_f @ gal_f.T
    nn = np.argpartition(-s, min(k, s.shape[1] - 1), axis=1)[:, :k]
    return np.array([Counter(gal_y[row]).most_common(1)[0][0] for row in nn])


def scores(y, p):
    classes = sorted(set(y))
    rec = {c: float(np.mean(p[y == c] == c)) for c in classes}
    f1 = []
    for c in classes:
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(y == p)), float(np.mean(f1)), rec


def confusion_png(y, p, classes, path, title):
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)))
    for a, b in zip(y, p):
        m[idx[a], idx[b]] += 1
    m = m / np.maximum(m.sum(1, keepdims=True), 1)
    cell = 22
    from PIL import ImageDraw
    pad = 210
    img = Image.new("RGB", (pad + cell * len(classes), 30 + pad + cell * len(classes)), "white")
    d = ImageDraw.Draw(img)
    d.text((5, 5), title, fill="black")
    for i, c in enumerate(classes):
        d.text((5, 30 + pad + i * cell + 5), c[:32], fill="black")
        for j in range(len(classes)):
            v = m[i, j]
            col = (int(255 * (1 - v)), int(255 * (1 - v * 0.6)), 255) if i == j else (255, int(255 * (1 - v)), int(255 * (1 - v)))
            d.rectangle([pad + j * cell, 30 + pad + i * cell, pad + (j + 1) * cell - 1, 30 + pad + (i + 1) * cell - 1], fill=col)
    for j, c in enumerate(classes):  # column labels, written vertically
        lbl = Image.new("RGB", (pad - 10, cell), "white")
        ImageDraw.Draw(lbl).text((2, 5), c[:32], fill="black")
        img.paste(lbl.rotate(90, expand=True), (pad + j * cell, 30))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", nargs="*", default=[str(MODELS / "RadioNet_v3_with_head.pth")],
                    help="one or more *_with_head.pth checkpoints to compare")
    ap.add_argument("--out", default=str(HERE / "report.md"))
    ap.add_argument("--domain-shift", nargs="*", default=[],
                    help="checkpoints trained on synthetic only: kNN gallery = all synthetic, queries = all real images")
    ap.add_argument("--cmap-shift", action="store_true", help="re-render Classic test images in every SDR++ colormap")
    ap.add_argument("--modes", nargs="*", default=["app", "full"])
    ap.add_argument("--no-current", action="store_true", help="skip the RadioNet v2 baseline")
    a = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    rows = load_splits()
    gal = [r for r in rows if r["split"] != "test"]
    qry = [r for r in rows if r["split"] == "test"]
    gal_y, q_y = np.array([r["class"] for r in gal]), np.array([r["class"] for r in qry])
    classes = sorted(set(r["class"] for r in rows))
    dev = device()

    models, sizes = {}, {}
    if not a.no_current:
        old = build_backbone().to(dev).eval()
        models["RadioNet v2"] = lambda x: (old(x), None)
    for path in a.v3:
        if not Path(path).exists():
            continue
        ck = torch.load(path, map_location="cpu")
        net = RadioNetV3(len(ck["classes"]), ckpt=None, arch=ck.get("arch", "resnet34"))
        net.load_state_dict(ck["model_state_dict"])
        net = net.to(dev).eval()
        assert ck["classes"] == classes, "class list changed since training - retrain"
        models[Path(path).stem.replace("_with_head", "")] = (lambda m: (lambda x: m(x)))(net)
        sizes[Path(path).stem.replace("_with_head", "")] = tuple(ck.get("input_size", (224, 224)))

    lines = ["# RadioNet evaluation (group-held-out test split)", "",
             f"gallery (train+val): {len(gal)} images, test queries: {len(qry)} images, {len(classes)} classes", "",
             "| model | preprocessing | method | accuracy | macro-F1 |", "|---|---|---|---|---|"]
    per_class = {}
    for name, fn in models.items():
        for mode in a.modes:
            sz = sizes.get(name, (224, 224))
            gf, _ = embed(fn, gal, mode, dev, size=sz)
            qf, ql = embed(fn, qry, mode, dev, size=sz)
            p = knn(gf, gal_y, qf)
            acc, f1, rec = scores(q_y, p)
            lines.append(f"| {name} | {mode} | kNN top-20 | {acc:.3f} | {f1:.3f} |")
            per_class[(name, mode, "kNN")] = rec
            other = np.array([colormap_of(r) not in ("", "Classic") for r in qry])
            if other.any():  # real captures rendered in non-Classic SDR++ colormaps
                acc_o, f1_o, _ = scores(q_y[other], p[other])
                lines.append(f"| {name} | {mode} | kNN, {int(other.sum())} captures in other colormaps | "
                             f"{acc_o:.3f} | {f1_o:.3f} |")
            if ql is not None:
                ph = np.array(classes)[ql.argmax(1)]
                acc, f1, rec = scores(q_y, ph)
                lines.append(f"| {name} | {mode} | classifier head | {acc:.3f} | {f1:.3f} |")
                per_class[(name, mode, "head")] = rec
                confusion_png(q_y, ph, classes, HERE / f"confusion_{name.replace(' ', '_')}_{mode}.png",
                              f"{name} {mode} head")
            confusion_png(q_y, p, classes, HERE / f"confusion_{name.replace(' ', '_')}_{mode}_knn.png",
                          f"{name} {mode} kNN")
            # cross-domain: synthetic gallery -> real queries, and real -> synthetic
            all_rows = gal + qry
            af = np.concatenate([gf, qf])
            ay = np.concatenate([gal_y, q_y])
            src = np.array([r["source"] for r in all_rows])
            for g_src, q_src in (("synthetic", "real"), ("real", "synthetic")):
                gm = src == "synthetic" if g_src == "synthetic" else src != "synthetic"
                qm = ~gm
                shared = set(ay[gm]) & set(ay[qm])
                qm &= np.isin(ay, list(shared))
                gm &= np.isin(ay, list(shared))
                if qm.sum() and gm.sum():
                    pp = knn(af[gm], ay[gm], af[qm])
                    acc, f1, _ = scores(ay[qm], pp)
                    lines.append(f"| {name} | {mode} | kNN {g_src}->{q_src} ({len(shared)} shared classes) | {acc:.3f} | {f1:.3f} |")
    # colormap shift: the same held-out waterfalls, rendered in each SDR++ colormap
    cmap_rows, cmap_lines = [], []
    if a.cmap_shift:
        from colormaps import gray_as, luts
        cq = [r for r in qry if colormap_of(r)]
        cy = np.array([r["class"] for r in cq])
        for name, fn in models.items():
            gf, _ = embed(fn, gal, "full", dev, size=sizes.get(name, (224, 224)))
            res = {}
            for cm in luts():
                qf, ql = embed(fn, cq, "full", dev, load=lambda r, cm=cm: gray_as(cached_level(r["file"], colormap_of(r)), cm),
                               size=sizes.get(name, (224, 224)))
                res[cm] = (scores(cy, knn(gf, gal_y, qf))[:2],
                           scores(cy, np.array(classes)[ql.argmax(1)])[:2] if ql is not None else None)
            cmap_rows.append((name, res))
        cmap_lines += ["", f"## Colormap shift ({len(cq)} SDR++-rendered test images, full preprocessing, kNN macro-F1"
                      " / head macro-F1)", "",
                  "| model | " + " | ".join(luts()) + " | mean |", "|---|" + "---|" * (len(luts()) + 1)]
        for name, res in cmap_rows:
            cells = [f"{v[0][1]:.2f}" + (f" / {v[1][1]:.2f}" if v[1] else "") for v in res.values()]
            mean = np.mean([v[0][1] for v in res.values()])
            cmap_lines.append(f"| {name} | " + " | ".join(cells) + f" | {mean:.2f} |")

    # true domain shift: models that never saw a real image
    for path in a.domain_shift:
        ck = torch.load(path, map_location="cpu")
        net = RadioNetV3(len(ck["classes"]), ckpt=None, arch=ck.get("arch", "resnet34"))
        net.load_state_dict(ck["model_state_dict"])
        net = net.to(dev).eval()
        syn = [r for r in rows if r["source"] == "synthetic"]
        real = [r for r in rows if r["source"] != "synthetic" and r["class"] in set(ck["classes"])]
        for mode in a.modes:
            gf, _ = embed(lambda x: net(x), syn, mode, dev)
            qf, ql = embed(lambda x: net(x), real, mode, dev)
            ry = np.array([r["class"] for r in real])
            acc, f1, _ = scores(ry, knn(gf, np.array([r["class"] for r in syn]), qf))
            lines.append(f"| {Path(path).stem.replace('_with_head', '')} (synthetic-only training) | {mode} | "
                         f"kNN synthetic gallery -> {len(real)} real images | {acc:.3f} | {f1:.3f} |")
            acc, f1, _ = scores(ry, np.array(ck["classes"])[ql.argmax(1)])
            lines.append(f"| {Path(path).stem.replace('_with_head', '')} (synthetic-only training) | {mode} | "
                         f"classifier head -> real images | {acc:.3f} | {f1:.3f} |")
        for src in ("real-old", "real-ota"):
            m = np.array([r["source"] == src for r in real])
            if m.any():
                qf_full, _ = embed(lambda x: net(x), [r for r, k in zip(real, m) if k], "full", dev)
                gf_full, _ = embed(lambda x: net(x), syn, "full", dev)
                acc, f1, _ = scores(np.array([r["class"] for r, k in zip(real, m) if k]),
                                    knn(gf_full, np.array([r["class"] for r in syn]), qf_full))
                lines.append(f"| {Path(path).stem.replace('_with_head', '')} (synthetic-only training) | full | "
                             f"kNN synthetic gallery -> {src} only ({int(m.sum())}) | {acc:.3f} | {f1:.3f} |")

    # style-leak baseline
    def style(r):
        im = Image.open(TMP / r["file"]).convert("RGB")
        w, h = im.size
        x = np.asarray(im.resize((8, 8)), np.float32).reshape(-1, 3)
        return np.r_[np.log(w), np.log(h), np.log(w / h), x.mean(0) / 255, x.std(0) / 255]
    sg, sq = np.array([style(r) for r in gal]), np.array([style(r) for r in qry])
    mu, sd = sg.mean(0), sg.std(0) + 1e-6
    sg, sq = (sg - mu) / sd, (sq - mu) / sd
    sg /= np.linalg.norm(sg, axis=1, keepdims=True)
    sq /= np.linalg.norm(sq, axis=1, keepdims=True)
    acc, f1, _ = scores(q_y, knn(sg, gal_y, sq))
    lines.append(f"| style-only baseline | - | kNN on size/aspect/colour | {acc:.3f} | {f1:.3f} |")

    lines += cmap_lines
    keys = list(per_class)
    lines += ["", "## Per-class recall on the test split", "",
              "| class | n test | " + " | ".join(f"{k[0]} {k[1]} {k[2]}" for k in keys) + " |",
              "|---|---|" + "---|" * len(keys)]
    for c in classes:
        n = int(np.sum(q_y == c))
        lines.append(f"| {c} | {n} | " + " | ".join(f"{per_class[k].get(c, float('nan')):.2f}" for k in keys) + " |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
