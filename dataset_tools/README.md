# dataset_tools

Scripts that build, curate and evaluate the Classidyne waterfall dataset and train RadioNet. The full how-to is
[`docs/DATASET_GUIDE.md`](../docs/DATASET_GUIDE.md); the dataset itself is described in
[`docs/DATASET_V3.md`](../docs/DATASET_V3.md).

| Path | What |
|---|---|
| `paths.py` | Shared locations. Image paths in the manifest are relative to the repo root (`datasets/waterfall/<class>/<sha256>.png`). |
| `manifest.csv` | One row per image: class, source (`synthetic` / `real-old` / `real-ota`), capture group, radio + SDR++ settings, colormap, generator parameters |
| `splits.csv`, `splits_random.csv` | Internal evaluation splits: group-held-out (whole capture groups) and per-image random. The published dataset has no split |
| `fft_manifest.csv` | The SDR++ spectrum-plot images in `datasets/fft/` (one per class) |
| `gen/` | NumPy signal generators (`signals.py`), DSP helpers, offline previews, SDR++ colormap LUTs (`colormaps.py`) |
| `capture/` | HackRF TX -> RTL-SDR -> SDR++ screenshot pipeline (`run_capture.py`, `run_full.py`), live receive-only captures (`run_live.py`), SDR++ control |
| `curate/` | Legacy import, class merges, balancing, live-frame occupancy check |
| `esp32_traffic/` | ESP32 firmware that generates Wi-Fi / BLE traffic for live captures |
| `train/` | RadioNet training (`train.py`) and model comparison (`evaluate.py`) |
| `eval/` | `app_eval.py` (evaluates the running app + its vector DB), `contact.py` (contact sheets) |

Large or machine-specific output (IQ files, raw screenshots, archived frames, image caches, trained models,
reports) goes to `tmp/dataset/`, which is git-ignored. `capture/winlist` is built with
`swiftc capture/winlist.swift -o capture/winlist`.

Common commands (repo root, `venv-classidyne` active):

```bash
python dataset_tools/capture/run_full.py --target 150                        # balanced bench capture
python dataset_tools/capture/run_full.py --colormap random --hires --target 20 # hi-res captures in other colormaps
python dataset_tools/train/train.py --split random --arch efficientnet_b0 --supcon 0.5 --cmap-aug 0.5 --input-size 448x224 --tag eval
python dataset_tools/train/evaluate.py --split random --v3 tmp/dataset/models/RadioNet_eval_with_head.pth --modes full --cmap-shift
python dataset_tools/train/train.py --all-data --arch efficientnet_b0 --supcon 0.5 --cmap-aug 0.5 --input-size 448x224 --tag release
python dataset_tools/eval/app_eval.py                                          # after embedding, server running
```
