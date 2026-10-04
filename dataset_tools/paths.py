"""Shared locations for the dataset tools.

Manifest and split files store image paths relative to the repository root ("datasets/waterfall/<class>/<sha256>.png"),
so they resolve against ROOT. Large, machine-specific intermediates go to SCRATCH (git-ignored).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "dataset_tools"
DATA = ROOT / "datasets" / "waterfall"
MANIFEST = TOOLS / "manifest.csv"
SPLITS = TOOLS / "splits.csv"                # group-held-out split (internal evaluation)
SPLITS_RANDOM = TOOLS / "splits_random.csv"  # per-image random split (internal evaluation)
SCRATCH = ROOT / "tmp" / "dataset"          # IQ files, raw screenshots, archived frames, caches, trained models
