# Classidyne app evaluation

> Measured with the split-trained model `RadioNet_v3_final_448.pth` (same recipe as the released all-data `RadioNet/RadioNet.pth`, which has seen every image and so cannot be scored on held-out data). Regenerate with `dataset_tools/eval/app_eval.py`.

Model: `efficientnet_b0`, preprocessing: whole frame 448x224. Waterfall collection: 4051 images. Voting exactly as `/api/classify` (top-20, similarity >= 0.5).

## 1. Held-out test images against the live vector DB (own capture group excluded)

| queries | accuracy | macro-F1 | no match | mean confidence (right / wrong) |
|---|---|---|---|---|
| 630 | 0.868 | 0.873 | 3 | 96% / 72% |

| source | queries | accuracy | macro-F1 |
|---|---|---|---|
| real-old | 137 | 0.781 | 0.709 |
| real-ota | 39 | 0.949 | 0.958 |
| synthetic | 454 | 0.888 | 0.851 |

Accuracy by reported confidence (how far the top-class percentage can be trusted):

| confidence | share of queries | accuracy |
|---|---|---|
| 0-50% | 4% | 0.286 |
| 50-75% | 5% | 0.559 |
| 75-90% | 7% | 0.643 |
| 90-100% | 83% | 0.937 |

Most frequent confusions (true -> predicted):

- sstv -> automatic-picture-transmission: 15
- packet -> sstv: 5
- RS41-Radiosonde -> sstv: 3
- lora -> bluetooth: 3
- 2ASK -> psk-qam: 2
- 2ASK -> OOK: 2
- am -> OOK: 2
- am -> airband: 2
- am -> 4FSK: 2
- bluetooth -> wifi: 2

| class | test queries | recall |
|---|---|---|
| 2ASK | 26 | 0.85 |
| 2FSK | 26 | 0.73 |
| 4FSK | 26 | 0.96 |
| OOK | 26 | 0.85 |
| RS41-Radiosonde | 26 | 0.85 |
| Radioteletype | 26 | 0.96 |
| ads-b | 25 | 1.00 |
| airband | 26 | 0.92 |
| ais | 26 | 0.96 |
| am | 26 | 0.73 |
| automatic-picture-transmission | 26 | 0.92 |
| bluetooth | 29 | 0.90 |
| cellular | 25 | 0.84 |
| digital-audio-broadcasting | 26 | 0.92 |
| digital-speech-decoder | 26 | 0.77 |
| fm | 26 | 1.00 |
| lora | 26 | 0.77 |
| packet | 26 | 0.65 |
| pocsag | 26 | 0.92 |
| psk-qam | 26 | 0.96 |
| sstv | 32 | 0.53 |
| vor | 26 | 0.96 |
| wifi | 25 | 1.00 |
| z-wave | 26 | 0.96 |

## 2. Colormap robustness (493 SDR++ test captures re-rendered per colormap)

| colormap | accuracy | macro-F1 |
|---|---|---|
| Classic | 0.874 | 0.832 |
| Classic Green | 0.878 | 0.836 |
| Electric | 0.878 | 0.844 |
| GQRX | 0.860 | 0.827 |
| Grey Scale | 0.897 | 0.856 |
| Inferno | 0.890 | 0.850 |
| Magma | 0.890 | 0.853 |
| Plasma | 0.886 | 0.858 |
| Smoke | 0.836 | 0.810 |
| Temper Colors | 0.892 | 0.848 |
| Turbo | 0.803 | 0.769 |
| Viridis | 0.880 | 0.855 |
| Vivid | 0.892 | 0.858 |
| WebSDR | 0.872 | 0.838 |
| **mean** | | **0.838** |

## 3. External images (not in the dataset)

| image | expected | top class | confidence | runner-up |
|---|---|---|---|---|
| `test_images/test1.png` | lora | digital-speech-decoder | 100% | - |
| `tests/lora.png` | lora | lora | 100% | - |
| `test_images/test2.png` | ? | fm | 95% | cellular (5%) |

## 4. HTTP (`http://localhost:5001`)

- `/api/stats`: `{'success': True, 'message': 'Stats fetched successfully.', 'embedding_status': 'Idle', 'waterfall_size': 4051, 'fft_size': 1}`
- `/api/classify`: 150/150 succeeded, median latency 372 ms (p90 429 ms)
- top class correct for 137/150 (includes the query's own capture group, so this is a sanity check, not an accuracy estimate)
