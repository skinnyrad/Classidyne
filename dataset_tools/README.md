# dataset_tools

Scripts that build, curate and evaluate the Classidyne waterfall dataset and train RadioNet. The full how-to is
[`docs/DATASET_GUIDE.md`](../docs/DATASET_GUIDE.md); the dataset itself is described in
[`docs/DATASET_V3.md`](../docs/DATASET_V3.md).

| Path | What |
|---|---|
| `paths.py` | Shared locations. Image paths in the manifest are relative to the repo root (`datasets/waterfall/<class>/<sha256>.png`). |
| `manifest.csv` | One row per image: class, source (`synthetic` / `real-old` / `real-ota`), capture group, radio + SDR++ settings, colormap, generator parameters |
| `splits.csv` | Group-held-out train / val / test split used for training and every evaluation |
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
python dataset_tools/train/train.py --arch efficientnet_b0 --supcon 0.5 --cmap-aug 0.5 --resplit --tag final
python dataset_tools/train/evaluate.py --v3 tmp/dataset/models/RadioNet_final_with_head.pth --modes full --cmap-shift
python dataset_tools/eval/app_eval.py                                          # after embedding, server running
```
