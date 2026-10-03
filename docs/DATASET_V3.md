# Classidyne Waterfall Dataset v3

Developer write-up for the rebuilt **waterfall-only** dataset (v3) and RadioNet v3. The dataset lives in `datasets/waterfall/`, every image is listed in [`dataset_tools/manifest.csv`](../dataset_tools/manifest.csv), and the scripts that built it are in `dataset_tools/`. For the full step-by-step process, see **[DATASET_GUIDE.md](DATASET_GUIDE.md)**; for how well the deployed app classifies, see **[APP_EVALUATION.md](APP_EVALUATION.md)**.

## What's in it

**4,051 images in 24 classes**, 141–172 per class (v2 ranged from 5 to 737). Every manifest row has the source, capture group, radio and SDR++ settings, colormap and generator parameters.

| Source | Images | What |
|---|---|---|
| `synthetic` | 2,892 | NumPy-generated signals sent with a HackRF in ISM bands (433 / 902–928 MHz, amp off), received on an RTL-SDR, rendered and screenshotted in **SDR++** (waterfall only) |
| `real-old` | 951 | Curated v2 images: duplicates/near-duplicates, UI borders, `unknown` and junk removed; each old capture session capped |
| `real-ota` | 208 | Verified live receive-only captures: Wi-Fi (ESP32 + ambient), BLE (ESP32, 2480 MHz), LTE/GSM downlinks, FM broadcast |

- **Resolution mix:** 1,056 images use a big FFT and fast waterfall (FFT 16k–65k at 500–3000 fps, like a 32k / 3000 fps SDR++ view of a 2.4 MHz span); the rest use per-class settings with the signal filling 8–60% of the span. 140 images are flagged `overloaded` (settings beyond what SDR++ renders in real time on the capture Mac).
- **Colormaps:** 2,609 Classic, plus **491 high-resolution captures in the 13 other SDR++ colormaps** (29–48 per colormap, 18–22 per class; bench and live). Legacy images come from other tools.
- **Zoom, FFT size, frame rate, gain, dB range, offset and signal parameters are randomized** per capture, so no single setting identifies a class.

### Class changes vs v2
- `on-off-keying` renamed **`OOK`**; `morse` + `remote-keyless-entry` merged into it (sub-type kept in the manifest).
- **New `2FSK`** class: continuous telemetry, ISM sensor packets and FSK keyfobs, 100 b/s–38.4 kb/s, 3–60 kHz deviation.
- `8PSK` + `16QAM` + `32QAM` (+ new BPSK, QPSK) → **`psk-qam`**. Constellations can't be seen on a waterfall.
- `unknown` → dropped.
- `drone-video`, `uav-video`, `hdmi`, `atsc` → **removed for now (planned for v3.5)**: too few images (5 / 6 / 29 / 67).
- `known_frequencies.json` updated for `OOK`, `psk-qam` and `2FSK`.

### Counts

| Class | n | Class | n | Class | n |
|---|---|---|---|---|---|
| 2ASK | 170 | 2FSK | 172 | 4FSK | 171 |
| ads-b | 166 | airband | 170 | ais | 172 |
| am | 169 | automatic-picture-transmission | 171 | bluetooth | 141 |
| cellular | 164 | digital-audio-broadcasting | 168 | digital-speech-decoder | 170 |
| fm | 171 | lora | 172 | OOK | 171 |
| packet | 169 | pocsag | 171 | psk-qam | 171 |
| Radioteletype | 169 | RS41-Radiosonde | 171 | sstv | 171 |
| vor | 172 | wifi | 170 | z-wave | 169 |

**Known gaps / v3.5 plans:**
- **Bring back drone-video, uav-video, hdmi and atsc** once there is enough live data. The analog 5.8 GHz FPV classes need a real transmitter.
- **Off-air airband / AIS / ADS-B / pager / VOR:** covered by bench and legacy images; the indoor antenna captured no usable off-air examples. An outdoor antenna would add them.
- The local `datasets/fft` folder has a single image; FFT views are not part of v3.

## RadioNet v3

**EfficientNet-B0, ImageNet start, cross-entropy + SupCon (0.5), colormap round-trip augmentation (0.5), whole-frame 448×224 input** (`RadioNet/RadioNet.pth`, 1280-d embeddings). The checkpoint stores `arch`, `preprocess` and `input_size`, which `app.py` reads.

### Results (group-held-out test split, 630 images)

Test images come from **capture groups never seen in training**: no frame of the same transmission or legacy session is in both train and test. "kNN" is exactly how Classidyne classifies (top-20 cosine vote).

| Model | kNN acc / macro-F1 | Head acc / macro-F1 | Real captures in other colormaps (59), kNN F1 | Colormap shift, mean of 14 maps |
|---|---|---|---|---|
| RadioNet v2 (ResNet-34, centre crop) | 0.50 / 0.49 | — | 0.17 | — |
| RadioNet v2, whole frame | 0.58 / 0.57 | — | 0.28 | 0.37 |
| v3 round 1 (pre-colormap captures, 224×224) | 0.83 / 0.83 | 0.80 / 0.80 | 0.55 | 0.81 |
| v3 final at 224×224 | 0.83 / 0.83 | 0.80 / 0.80 | 0.62 | 0.81 |
| **v3 final, 448×224 (deployed)** | **0.87 / 0.87** | **0.85 / 0.85** | **0.64** | **0.84** |
| style-only baseline (size / aspect / colour) | 0.27 / 0.26 | | | |

Through the live app and vector DB (`dataset_tools/eval/app_eval.py`): **0.868 accuracy / 0.873 macro-F1** on the held-out images, colormap-shift mean 0.84, `/api/classify` median latency ~0.38 s.

**Takeaways:**
1. **Whole-frame preprocessing.** The v2 centre-crop discarded about 60% of a wide waterfall's span.
2. **448×224 input matters for wide high-res views.** At 224 px a narrow signal in a 2.4 MHz span is 1–5 px wide: high-res frames whose signal fills < 3% of the span were only 37% correct (vs 93% for normal frames). Doubling the frequency resolution raised 2FSK from 0.31 to 0.73 recall, and RS41, VOR and APT by 0.1–0.2.
3. **Colormaps matter even in grayscale.** Most SDR++ maps are not monotonic in brightness (Classic: strong signals turn *dark* in grayscale). Colormap round-trip augmentation lifted the mean over all 14 maps from 0.71 to 0.81–0.84 macro-F1, and the real colormap captures improved the model on genuine non-Classic screenshots (0.55 → 0.64).
4. **EfficientNet-B0 + SupCon from ImageNet** beat ResNet-34 (ImageNet, SupCon or v2-RadioNet start) and training from scratch (see `DATASET_GUIDE.md` §13).
5. **Still weakest:** SSTV (0.53, mostly confused with APT: both are slow narrow FM audio images), packet (0.65), 2FSK and AM (0.73), LoRa and DSD (0.77). Legacy `real-old` images score lowest (macro-F1 0.71) because each comes from a session the model never saw. Of the external test images, `tests/lora.png` is still classified correctly, but the grey LoRa screenshot `test_images/test1.png` (another tool, no SDR++ colormap) now lands on digital-speech-decoder; the round-1 224×224 model got it right. More non-SDR++ real captures would help here.

Reports (git-ignored, `tmp/dataset/reports/`): `report_final_448.md`, `report_final.md`, `report_r1.md`, `app_eval_r1.md`; training logs in `tmp/dataset/logs/`; models in `tmp/dataset/models/`.

## Re-running

```bash
source venv-classidyne/bin/activate
python dataset_tools/train/train.py --arch efficientnet_b0 --supcon 0.5 --cmap-aug 0.5 --input-size 448x224 --tag v3_final_448
python dataset_tools/train/evaluate.py --v3 tmp/dataset/models/RadioNet_v3_final_448_with_head.pth --modes full --cmap-shift
cp tmp/dataset/models/RadioNet_v3_final_448.pth RadioNet/RadioNet.pth
rm -rf classidyne_db && python app.py      # then POST /api/start-embedding
python dataset_tools/eval/app_eval.py --port 5001
```

Kaggle: about 13 GB of full-resolution PNGs. Publish `dataset_tools/manifest.csv` and `dataset_tools/splits.csv` with it so others use group-held-out splits.

Archived (not deleted) frames are in `tmp/dataset/_*/`: pilots, rejected live frames, faint colormap frames, balancing overflow, merged-class overflow. The v2 waterfall dataset, v2 `RadioNet.pth` and v2 vector DB are in `tmp/backups/`.
