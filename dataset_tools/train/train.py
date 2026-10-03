"""Fine-tune RadioNet (ResNet-34) on dataset v3.

* starts from RadioNet/RadioNet.pth, adds a linear head, trains with class-balanced sampling
* group-held-out train/val/test split (dataset_tools/splits.csv) - no capture group is on both sides
* physics-safe augmentation: random time/frequency crops, brightness/contrast/gamma; no flips
  (mirroring would turn LoRa up-chirps into down-chirps, swap USB/LSB, etc.)
* saves the backbone in the format app.py loads: {"model_state_dict", "arch", "preprocess", "classes"}

usage: python train.py [--epochs 25] [--mode full|app] [--resplit]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import (MEAN, MODELS, OLD_CKPT, SPLITS, STD, RadioNetV3, cached_gray, cached_level, colormap_of, device,
                    load_splits, make_splits, tensorize)
from colormaps import gray_as, luts  # noqa: E402  (on the path via common)

OUT = MODELS


class WaterfallDS(Dataset):
    def __init__(self, rows, classes, train: bool, mode: str, cmap_aug: float = 0.0, size=(224, 224)):
        self.rows, self.idx, self.train, self.mode = rows, {c: i for i, c in enumerate(classes)}, train, mode
        self.cmap_aug, self.size = cmap_aug, tuple(size)
        aspect = size[0] / size[1]
        self.cmaps = list(luts()) if cmap_aug else []
        self.crop = T.RandomResizedCrop((size[1], size[0]), scale=(0.35, 1.0), ratio=(0.4 * aspect, 2.5 * aspect),
                                        interpolation=T.InterpolationMode.BICUBIC)
        self.jitter = T.ColorJitter(brightness=0.35, contrast=0.35)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        cmap = colormap_of(r)
        if self.train and self.cmap_aug and cmap:
            # crop the level image, then render it in a random SDR++ colormap (its own colormap otherwise)
            name = str(np.random.choice(self.cmaps)) if np.random.random() < self.cmap_aug else cmap
            img = gray_as(self.crop(cached_level(r["file"], cmap)), name)
            img = self.jitter(img.convert("RGB")).convert("L")
        elif self.train:
            img = self.jitter(self.crop(cached_gray(r["file"]).convert("RGB"))).convert("L")
        else:
            img = cached_gray(r["file"])
        if self.train:
            g = float(np.exp(np.random.uniform(-0.4, 0.4)))  # colormap/brightness differences between tools
            img = Image.fromarray((255 * (np.asarray(img, np.float32) / 255) ** g).astype(np.uint8))
            t = TF.normalize(TF.to_tensor(img.convert("RGB")), MEAN, STD)
        else:
            t = tensorize(img, self.mode, self.size)
        return t, self.idx[r["class"]]


def supcon_loss(feats, labels, temp=0.1):
    """Supervised contrastive loss (Khosla et al. 2020) on L2-normalised embeddings: pulls same-class
    embeddings together - exactly what Classidyne's cosine kNN lookup relies on."""
    z = torch.nn.functional.normalize(feats, dim=1)
    sim = z @ z.T / temp
    n = len(labels)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    sim = sim.masked_fill(eye, -1e9)
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    cnt = pos.sum(1)
    keep = cnt > 0
    return -((logprob * pos).sum(1)[keep] / cnt[keep]).mean()


def macro_f1(y, p, n):
    f1 = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        if tp + fn == 0:
            continue
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1))


@torch.no_grad()
def evaluate(model, loader, dev, n):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        _, logits = model(x.to(dev))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float((y == p).mean()), macro_f1(y, p, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--mode", default="full", choices=["full", "app"])
    ap.add_argument("--resplit", action="store_true")
    ap.add_argument("--extend-split", action="store_true",
                    help="keep existing group assignments, assign only groups new to the manifest")
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--init", default="imagenet", choices=["imagenet", "radionet", "scratch"])
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--supcon", type=float, default=0.0, help="weight of the SupCon loss (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input-size", default="224x224",
                    help="model input WIDTHxHEIGHT for --mode full; wider keeps narrow signals in wide spans visible")
    ap.add_argument("--cmap-aug", type=float, default=0.0,
                    help="probability of re-rendering a Classic image in another SDR++ colormap (0 = off)")
    ap.add_argument("--train-source", default="all", choices=["all", "synthetic"],
                    help="synthetic = train on bench captures only (domain-shift experiment)")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.input_size.lower().split("x"))
    rows = (make_splits(extend=a.extend_split) if a.resplit or a.extend_split or not SPLITS.exists()
            else load_splits())
    classes = sorted({r["class"] for r in rows})
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    if a.train_source == "synthetic":
        classes = sorted({r["class"] for r in rows if r["source"] == "synthetic"})
        tr = [r for r in rows if r["source"] == "synthetic" and r["split"] != "test"]
        va = [r for r in rows if r["source"] == "synthetic" and r["split"] == "test"]
    print(f"{len(classes)} classes | train {len(tr)} val {len(va)} test {sum(r['split'] == 'test' for r in rows)}")
    cnt = {c: sum(r["class"] == c for r in tr) for c in classes}
    weights = [1.0 / cnt[r["class"]] for r in tr]
    dl_tr = DataLoader(WaterfallDS(tr, classes, True, a.mode, a.cmap_aug, size), batch_size=a.bs,
                       sampler=WeightedRandomSampler(weights, len(tr), replacement=True), num_workers=4,
                       persistent_workers=True)
    dl_va = DataLoader(WaterfallDS(va, classes, False, a.mode, size=size), batch_size=64, num_workers=2)
    dev = device()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = RadioNetV3(len(classes), ckpt=OLD_CKPT if a.init == "radionet" else None, arch=a.arch,
                       imagenet=a.init == "imagenet").to(dev)
    print(f"init={a.init} arch={a.arch} supcon={a.supcon} mode={a.mode} cmap_aug={a.cmap_aug} input={size}")
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 1e-4},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 3e-3], total_steps=a.epochs * len(dl_tr),
                                                pct_start=0.15)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    OUT.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            feats, logits = model(x)
            loss = lossf(logits, y)
            if a.supcon:
                loss = loss + a.supcon * supcon_loss(feats, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        acc, f1 = evaluate(model, dl_va, dev, len(classes))
        log.append({"epoch": ep, "loss": tot / n, "val_acc": acc, "val_macro_f1": f1})
        print(f"ep {ep:2d} loss {tot / n:.3f} val acc {acc:.3f} macroF1 {f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best = f1
            torch.save({"model_state_dict": model.backbone.state_dict(), "arch": a.arch, "preprocess": a.mode, "input_size": list(size),
                        "classes": classes}, OUT / f"RadioNet_{a.tag}.pth")
            torch.save({"model_state_dict": model.state_dict(), "classes": classes, "mode": a.mode,
                        "arch": a.arch, "init": a.init, "supcon": a.supcon, "cmap_aug": a.cmap_aug,
                        "input_size": list(size)},
                       OUT / f"RadioNet_{a.tag}_with_head.pth")
    (OUT / f"train_{a.tag}.json").write_text(json.dumps({"best_val_macro_f1": best, "classes": classes,
                                                          "mode": a.mode, "init": a.init, "arch": a.arch,
                                                          "supcon": a.supcon, "log": log}, indent=2))
    print("best val macro-F1", best)


if __name__ == "__main__":
    main()
