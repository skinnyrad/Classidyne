"""Shared pieces for RadioNet v3 training/evaluation: splits, image cache, model, preprocessing."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "gen")]
from paths import MANIFEST, ROOT, SCRATCH, SPLITS, SPLITS_RANDOM  # noqa: E402

REPO = TMP = ROOT  # manifest paths are relative to the repo root
CACHE = SCRATCH / "train_cache"
MODELS = SCRATCH / "models"
REPORTS = SCRATCH / "reports"
# v2 RadioNet (ResNet-34) baseline: RadioNet/RadioNet.pth now holds v3, so the v2 weights are read from the backup
OLD_CKPT = next((p for p in (SCRATCH.parent / "backups" / "radionet_v2" / "RadioNet.pth",
                             REPO / "RadioNet" / "RadioNet.pth") if p.exists()), REPO / "RadioNet" / "RadioNet.pth")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # timm resnet34 defaults (same as app.py)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- splits

def make_splits(seed=0, val=0.15, test=0.15, extend=False) -> list[dict]:
    """Group-held-out split: every capture group (one TX variant / one legacy screenshot session /
    one live centre frequency) lands entirely in train, val or test, per class.
    extend=True keeps the split of every group already in splits.csv and only assigns new groups, so models
    trained on the old split can still be compared fairly on the new test set."""
    rows = list(csv.DictReader(open(MANIFEST)))
    rng = np.random.default_rng(seed)
    prior = {r["group_id"]: r["split"] for r in load_splits()} if extend and SPLITS.exists() else {}
    by_cls: dict[str, dict[str, list]] = {}
    for r in rows:
        by_cls.setdefault(r["class"], {}).setdefault(r["group_id"], []).append(r)
    out = []
    for cls, groups in by_cls.items():
        keys = list(groups)
        rng.shuffle(keys)
        keys.sort(key=lambda k: k not in prior)  # already-assigned groups first (stable: shuffle order kept)
        n = sum(len(v) for v in groups.values())
        tgt = {"test": test * n, "val": val * n}
        got = {"test": 0, "val": 0}
        for k in keys:
            split = "train"
            if k in prior:
                split = prior[k]
            elif len(keys) >= 3:
                for s in ("test", "val"):
                    if got[s] < tgt[s] and (got[s] == 0 or got[s] + len(groups[k]) <= tgt[s] * 1.5):
                        split = s
                        break
            if split != "train":
                got[split] += len(groups[k])
            for r in groups[k]:
                out.append({"file": r["file"], "class": cls, "source": r["source"], "group_id": k, "split": split,
                            "subtype": json.loads(r["params"] or "{}").get("subtype", ""),
                            "colormap": colormap_of(r)})
    with open(SPLITS, "w", newline="") as f:
        w = csv.DictWriter(f, out[0].keys())
        w.writeheader()
        w.writerows(out)
    return out


def make_random_splits(seed=0, val=0.15, test=0.15) -> list[dict]:
    """Per-image random split, stratified by class (the common Kaggle-style split). Frames of one capture group
    can land on both sides, so scores measure performance on data like the training captures."""
    rows = list(csv.DictReader(open(MANIFEST)))
    rng = np.random.default_rng(seed)
    out = []
    for cls in sorted({r["class"] for r in rows}):
        rs = [r for r in rows if r["class"] == cls]
        order = rng.permutation(len(rs))
        n_test, n_val = round(test * len(rs)), round(val * len(rs))
        for k, i in enumerate(order):
            r = rs[i]
            split = "test" if k < n_test else "val" if k < n_test + n_val else "train"
            out.append({"file": r["file"], "class": cls, "source": r["source"], "group_id": r["group_id"],
                        "split": split, "subtype": json.loads(r["params"] or "{}").get("subtype", ""),
                        "colormap": colormap_of(r)})
    with open(SPLITS_RANDOM, "w", newline="") as f:
        w = csv.DictWriter(f, out[0].keys())
        w.writeheader()
        w.writerows(out)
    return out


def load_splits(kind: str = "group") -> list[dict]:
    return list(csv.DictReader(open(SPLITS_RANDOM if kind == "random" else SPLITS)))


def all_rows() -> list[dict]:
    """Every manifest image, for training the release model on all the data."""
    return [{"file": r["file"], "class": r["class"], "source": r["source"], "group_id": r["group_id"],
             "split": "train", "colormap": colormap_of(r)} for r in csv.DictReader(open(MANIFEST))]


# ----------------------------------------------------------------------------- image cache

def cached_gray(rel: str, max_side=640) -> Image.Image:
    """Grayscale (as app.py does: convert('L')) downsized copy, cached on disk for fast epochs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / (hashlib.md5(rel.encode()).hexdigest() + ".png")
    if key.exists():
        return Image.open(key)
    img = Image.open(TMP / rel).convert("L")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(key)
    return img


def colormap_of(r: dict) -> str:
    """SDR++ colormap an image was rendered with ('' = unknown, e.g. legacy screenshots from other tools).
    Bench/live captures from before the colormap column existed are all Classic."""
    if r.get("colormap"):
        return r["colormap"]
    return "Classic" if r["source"] in ("synthetic", "real-ota") else ""


def cached_level(rel: str, cmap: str = "Classic", max_side=640) -> Image.Image:
    """Colormap-index ('signal level') image of a waterfall rendered in SDR++ colormap ``cmap``."""
    from colormaps import to_level
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / ("lv_" + hashlib.md5(rel.encode()).hexdigest() + ".png")
    if key.exists():
        return Image.open(key)
    img = Image.open(TMP / rel).convert("RGB")
    img.thumbnail((max_side, max_side), Image.NEAREST)  # no blending across colormap stops
    lv = to_level(img, cmap)
    lv.save(key)
    return lv


# ----------------------------------------------------------------------------- model

def build_backbone(ckpt: Path | None = OLD_CKPT, arch="resnet34", imagenet=False) -> torch.nn.Module:
    """Feature extractor as app.RadioNetExtractor builds it (resnet34 -> 512-d pooled output).
    ckpt=OLD_CKPT: current RadioNet weights; imagenet=True: timm ImageNet weights; neither: random init."""
    m = timm.create_model(arch, pretrained=imagenet, num_classes=0, global_pool="avg")
    if ckpt is not None and not imagenet:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        sd = {k: v for k, v in sd.items() if "fc" not in k}
        m.load_state_dict(sd, strict=False)
    return m


class RadioNetV3(torch.nn.Module):
    def __init__(self, n_classes: int, ckpt: Path | None = OLD_CKPT, arch="resnet34", imagenet=False):
        super().__init__()
        self.backbone = build_backbone(ckpt, arch, imagenet)
        self.head = torch.nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        f = self.backbone(x)
        return f, self.head(f)


# ----------------------------------------------------------------------------- preprocessing

def tensorize(img: Image.Image, mode: str, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """mode='app'   : timm resnet34 eval transform (resize 248 short side + centre crop 224) = v2 app.py
       mode='full'  : whole waterfall squashed to size = (width, height), keeping the full span and time window"""
    import torchvision.transforms.functional as TF
    img = img.convert("L").convert("RGB")
    if mode == "app":
        return _app_transform()(img)
    else:
        img = img.resize(tuple(size), Image.BICUBIC)
    t = TF.to_tensor(img)
    return TF.normalize(t, MEAN, STD)


_APP_TF = None


def _app_transform():
    """The exact eval transform app.RadioNetExtractor uses."""
    global _APP_TF
    if _APP_TF is None:
        from timm.data.config import resolve_data_config
        from timm.data.transforms_factory import create_transform
        _APP_TF = create_transform(**resolve_data_config({}, model="resnet34"))
    return _APP_TF
