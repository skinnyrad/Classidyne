# Building a Classidyne Waterfall Dataset

This guide walks through building the Classidyne v3 waterfall dataset (24 classes, 4,051 images), and is written so you can repeat the process. It covers:

- generating signals with NumPy;
- transmitting them with a HackRF;
- capturing them as SDR++ waterfalls on an RTL-SDR;
- recording real Wi-Fi and Bluetooth from an ESP32;
- curating the legacy images;
- balancing the classes;
- training and evaluating RadioNet.

The tooling lives in `dataset_tools/`; large, machine-specific intermediates go to the git-ignored `tmp/dataset/` (see [File map](#15-file-map)). Paths below are relative to the repo root unless noted otherwise. For a summary of the dataset itself (contents, class changes, model results, merge checklist), see [DATASET_V3.md](DATASET_V3.md).

> **TL;DR pipeline**
>
> ```
> NumPy generator ─► .cs8 file ─► hackrf_transfer -R (433/915 MHz ISM, amp off)
>                                        │  (over the air, short bench distance)
>                                        ▼
>                          RTL-SDR ─► SDR++ (driven by config + rigctl) ─► window screenshot
>                                        ▼
>                     crop waterfall ─► PNG (sha256 name) ─► manifest.csv row
>                                        ▼
>        curate legacy + live captures ─► balance ─► group-held-out split ─► train/evaluate RadioNet
> ```

---

## Contents

1. [Why v3: what was wrong with the old dataset](#1-why-v3)
2. [Hardware and software](#2-hardware-and-software)
3. [macOS permissions and one-time setup](#3-macos-permissions-and-one-time-setup)
4. [RF safety and legality](#4-rf-safety-and-legality)
5. [Generating signals with NumPy](#5-generating-signals-with-numpy)
6. [From baseband to the air: the HackRF](#6-from-baseband-to-the-air)
7. [Driving SDR++ without a mouse](#7-driving-sdr-without-a-mouse)
8. [Calibration: scroll speed, frequency error, brightness](#8-calibration)
9. [The bench capture loop](#9-the-bench-capture-loop)
10. [Live captures: ESP32, HackRF RX, off-air](#10-live-captures)
11. [Curating the legacy dataset](#11-curating-the-legacy-dataset)
12. [Class design decisions and balancing](#12-class-design-and-balancing)
13. [Training and evaluating RadioNet](#13-training-and-evaluating-radionet)
14. [Troubleshooting (everything that broke)](#14-troubleshooting)
15. [File map](#15-file-map)
16. [Merging into `datasets/` and publishing](#16-merging-and-publishing)

---

## 1. Why v3

The v2 Kaggle dataset (`halcy0nic/radio-frequecy-rf-signal-image-classification`) had four structural problems. Every design decision below exists to fix one of them.

| Problem | Evidence | Fix in v3 |
|---|---|---|
| **Severe imbalance** | 5 images (2ASK, 4FSK, drone-video) vs. 737 (fm), about 147× | Synthesize the rare classes on the bench and cap each class at ~150 |
| **The class could be guessed from presentation** | Each class came mostly from one capture session with its own colormap, software and screenshot size (e.g. 273/299 `packet` images were 945×468 in a yellow palette) | One capture tool (SDR++), one crop rule, randomized display settings; mostly Classic, plus ~20 high-res captures per class spread over the 13 other SDR++ colormaps; legacy images capped per session |
| **Near-duplicates** | Consecutive screenshots of the same transmission | Perceptual-hash dedup, plus **group-held-out** evaluation so frames of one transmission never sit in both train and test |
| **Data quality** | `unknown` class, UI chrome in screenshots, `.DS_Store`, classes with fewer than 10 images | Drop `unknown`, trim UI borders, quality filters, label checks on live captures |

Third-party notebooks reported about 98% accuracy and per-class F1 = 1.0 on v2. That was mostly memorization: the splits were random per image, the augmentations included horizontal/vertical flips (which mirror frequency and reverse time), and the RGB colormaps leaked the class. The v3 evaluation is designed so those shortcuts don't count (§13).

---

## 2. Hardware and software

| Item | Role | Notes |
|---|---|---|
| **HackRF One** (firmware 2.4.0) | Transmitter for bench signals; receiver (up to 20 MS/s) for wideband live captures | `brew install hackrf` |
| **RTL-SDR Blog V4** | Receiver for bench captures and narrow off-air signals | about 500 kHz–1.766 GHz, up to 2.4 MS/s usable |
| **ESP32 dev board** (CP2102, `/dev/cu.usbserial-0001`) | Real Wi-Fi and BLE traffic source | Flashed with `dataset_tools/esp32_traffic` |
| **SDR++** 1.3.0 (`/Applications/SDR++.app`) | Renders every waterfall | Config in `~/Library/Application Support/sdrpp/` |
| Python 3.14 venv `venv-classidyne` | numpy, scipy, pillow, torch, timm, pyserial | `pip install -r requirements.txt pyserial` |
| `arduino-cli` + `esp32:esp32` core 3.3.11 | ESP32 build and flash | |
| Xcode command-line tools | `swiftc` for the two tiny window helpers | |

Antennas: a short antenna on each radio, about 1 m apart on the bench. The HackRF needs something usable at both 433/915 MHz (TX) and 2.4 GHz (RX).

---

## 3. macOS permissions and one-time setup

1. **Screen & System Audio Recording → Claude / your terminal.** Required: every image is a window screenshot (`screencapture -l <windowID>`). Without it you get `could not create image from window`.
2. **Keep the screen awake and unlocked.**
   - The capture scripts run under `caffeinate -dims` and declare user activity (`caffeinate -u -t 2`) every session, so the screensaver and lock never start.
   - A locked screen would produce black screenshots. They are rejected by the signal check, but auto-levelling would cache bad levels.
3. **Accessibility is not needed.** SDR++ is controlled through its config files and rigctl server, not mouse clicks (§7).
4. **Build the window helper:**

   ```bash
   cd dataset_tools/capture && swiftc -O winlist.swift -o winlist
   ```

   `winlist` prints `<windowID> x y w h` for the largest SDR++ window, using CoreGraphics.
5. **Back up the SDR++ config before touching it:**

   ```bash
   cp ~/Library/Application\ Support/sdrpp/*.json tmp/backups/sdrpp/
   ```

   Restore it the same way when you finish.

---

## 4. RF safety and legality

These rules are enforced in code (`dataset_tools/capture/run_capture.py`) as well as by convention.

- **Transmit only inside ISM bands:**
  - **433.05–434.79 MHz:** quiet here, but only 1.74 MHz wide.
  - **902–928 MHz:** wide, but busy with local meters, sensors and LoRa.

  `band_of()` refuses anything else, and `Transmitter.start()` asserts the LO is near an ISM band.
- **Never transmit on a protocol's real frequency.** ADS-B is not sent at 1090 MHz, AIS not at 161.975/162.025 MHz, and the same goes for airband, VOR, pagers, cellular and broadcast. A waterfall image has no frequency axis, so the in-band shape is all that matters.
- **The whole occupied bandwidth must fit inside the band.** Each variant's 99.9%-power bandwidth is measured. If `center ± obw/2` would cross a band edge, the offset is clipped. If the signal can't fit at all, the variant is skipped. Wide signals (over 1.6 MHz) go to 902–928 MHz; 0.9–1.6 MHz signals are centred at 433.92 MHz.
- **HackRF amp off (`-a 0`), TX VGA 0–30 dB.** The two antennas are about a metre apart, so tiny power is plenty.
- **Always stop the radio.** `hackrf_transfer -R` loops forever. `Transmitter` registers `atexit` and SIGTERM/SIGINT handlers that stop it, plus a final `pkill -INT -x hackrf_transfer`. One early run killed mid-transmission left an orphaned `hackrf_transfer` running for minutes. After any crash, check:

  ```bash
  pgrep -lf hackrf_transfer
  ```

- **Real-world signals are receive-only.** Live cellular, ATSC, FM, HDMI leakage and 2.4 GHz captures transmit nothing. The ESP32 is a normal low-power (8 dBm) Wi-Fi/BLE device on its own SSID `CLASSIDYNE-TEST`, and never impersonates another network.

---

## 5. Generating signals with NumPy

All generators are in **`dataset_tools/gen/signals.py`**. They build on the approach in *Generating Signals with NumPy* (`aft-rfctf/tmp/generating-signals-with-numpy/article.md`) and the helpers in `aft-rfctf/iq_recordings/iqlib.py`, which is copied to `dataset_tools/gen/iqlib.py`.

### 5.1 The contract

```python
def gen_<class>(rng, fs, dur) -> (iq: complex64 ndarray at baseband (0 Hz), meta: dict)
```

- `rng` is a seeded `np.random.Generator`, so every image is reproducible from the manifest's seed.
- `fs` is the generator sample rate (the profile's `gen_fs`): 250 kS/s for narrow modes, 1–2.4 MS/s for wide ones.
- `meta` holds every randomized parameter (symbol rate, deviation, SF/BW, WPM...). It ends up in the manifest's `params` column.
- The RF offset is **not** applied in the generator. The capture loop picks it after measuring bandwidth (§9).

`GENERATORS` maps sub-type to function. `CLASS_OF` and `SUBTYPES` map sub-types onto dataset classes:

- `morse` and `remote-keyless-entry` → `OOK`
- `BPSK`, `QPSK`, `8PSK`, `16QAM`, `32QAM` → `psk-qam`

### 5.2 Shaping rules (the difference between "looks synthetic" and "looks real")

1. **No hard amplitude edges.** On/off envelopes are smoothed with a Hann kernel: `shape_edges()`, raised-cosine ramps of 2–8 ms for CW and about 50 µs for OOK. Every burst gets `ramp()` power-up and power-down. Hard edges cause *key clicks*: wide horizontal splatter lines across the whole waterfall.
2. **No phase jumps in FSK.** Build the instantaneous-frequency track and integrate it, `cpfsk(freq) = exp(j·2π·cumsum(freq)/fs)`. Gaussian filtering of the symbol track (`symbols_to_track(..., bt)`) gives GFSK/GMSK.
3. **Linear modulations are RRC-shaped.** `rrc_linear()` upsamples the symbols and convolves with `iqlib.rrc_taps(beta, sps, 10)`. β is randomized from 0.2 to 0.5.
4. **Long convolutions use `scipy.signal.oaconvolve`.** A full-length FFT convolution of a 20 s, 2.4 MS/s signal exhausted memory and the process was killed (exit 137).
5. **OFDM is resampled with `resample_poly`.** Linear interpolation creates spectral images at the band edges.

### 5.3 What each generator models

| Class | Model (parameters randomized per variant) |
|---|---|
| OOK | Three sub-types:<br>• OOK packets: preamble + sync + random payload, 300 b/s–8 kb/s, optional Manchester, 2–5 repeats<br>• Morse: 8–30 WPM, random text<br>• Keyfob PWM-OOK: KeeLoq-style preamble + 66-bit word, 3–7 repeats |
| 2ASK | Two-level ASK, 1–50 kBd, low level 0–0.4, continuous or bursty |
| 2FSK | Binary CPFSK, 100 b/s–38.4 kb/s (low rates favoured), deviation 3–60 kHz, optional Gaussian BT 0.5/1.0; continuous telemetry, sensor packets (preamble + sync + payload) or keyfob words repeated 3–7× |
| 4FSK | CPFSK ±1/±3, 1.2–19.2 kBd, optional Gaussian |
| psk-qam | BPSK / QPSK / 8PSK / 16QAM / 32QAM (cross), 25–500 kBd, RRC β 0.2–0.5, continuous or bursty |
| am | Broadcast AM (music/speech, 4.5–9 kHz audio, depth 0.4–0.95) |
| airband | Push-to-talk AM voice: carrier keys on for transmissions |
| fm | WBFM multiplex: mono + 19 kHz pilot + 38 kHz L–R + 57 kHz RDS, 75 kHz deviation |
| pocsag | 2FSK ±4.5 kHz, 512/1200/2400 Bd, 576-bit preamble + batches |
| ads-b | Mode S PPM (8 µs preamble, 56/112 bits), random aircraft levels, band-limited to 1.5 MHz |
| ais | GMSK 9600 b/s BT 0.4, 26.7 ms SOTDMA bursts on two channels 50 kHz apart |
| packet | AX.25/APRS: AFSK1200 audio on NBFM |
| Radioteletype | FSK 45.45/50/75 Bd, 85–850 Hz shift |
| sstv | Martin/Scottie-like lines: 1200 Hz sync + 1500–2300 Hz pixels on NBFM |
| automatic-picture-transmission | NOAA APT: 2400 Hz AM subcarrier, sync A/B, FM ±17 kHz |
| lora | CSS chirps: SF chosen so a symbol is ≥ 3 ms (BW 125/250/500 kHz), preamble + SFD + payload |
| RS41-Radiosonde | GFSK 4800 Bd, ~0.8 s frames per second |
| z-wave | R1 9.6k Manchester FSK, R2 40k FSK, R3 100k GFSK; frame + ACK |
| digital-speech-decoder | DMR (4FSK 4800 sym/s, 30 ms TDMA) or P25 C4FM, with PTT overs |
| vor | AM carrier + 30 Hz AM + 9960 Hz subcarrier (±480 Hz FM at 30 Hz) + 1020 Hz Morse ident |
| digital-audio-broadcasting | DAB Mode I OFDM: 1536 × 1 kHz carriers, guard 246 µs, null symbol every 96 ms |
| cellular | GSM (GMSK 270.833 kBd, multiple carriers, TDMA slots) or LTE 1.4 MHz (72 subcarriers, per-subframe RB allocation) |

Check generators offline before going on air:

```bash
cd dataset_tools/gen && python preview.py            # -> tmp/dataset/preview/<class>.png + _contact.png
```

---

## 6. From baseband to the air

**`write_cs8(iq, gen_fs, tx_fs, shift_hz, path)`** in `run_capture.py`:

1. resamples (`resample_poly`) from `gen_fs` up to the HackRF rate `tx_fs`;
2. multiplies by `exp(j·2π·shift·n/tx_fs)` to place the signal;
3. normalizes to a peak of 100 (out of 127 in int8, leaving headroom against clipping);
4. writes interleaved `I0 Q0 I1 Q1...` as int8, the **CS8** format `hackrf_transfer -t` expects.

**Keep the HackRF LO out of the picture.** The HackRF has a DC/LO-leakage spike at its tuned frequency. `tx_plan(span, offset)` tunes the HackRF *outside* the RTL's visible span and moves the signal back into view digitally:

| Visible span | HackRF rate | LO position |
|---|---|---|
| ≤ 300 kHz | 2 MS/s | span/2 + 250 kHz |
| ≤ 1.1 MHz | 4 MS/s | span/2 + 400 kHz |
| wider | 8 MS/s | span/2 + 1 MHz |

```bash
hackrf_transfer -t sig.cs8 -f <LO Hz> -s <tx_fs> -x <0-30> -a 0 -R
```

Files are generated just before each transmission and deleted right after (`tmp/dataset/iq/` stays small). Each one covers about 1.2 screens of signal, 2–14 s, and `-R` loops it.

---

## 7. Driving SDR++ without a mouse

SDR++ reads display settings only at startup. The control layer (`dataset_tools/capture/sdrpp_ctl.py`) therefore uses a **restart-to-reconfigure** pattern, with **rigctl** for runtime control.

### 7.1 Restart with a profile

`restart_with(profile)` does:

1. `osascript -e 'quit app "SDR++"'`. SDR++ rewrites its config on exit, so edit only while it's closed.
2. `apply_profile()` edits:
   - `config.json`:
     - source, `frequency`, `fftSize`, `fftRate`, `colorMap="Classic"`, `min`/`max` dB, `decimation`;
     - layout: `showMenu=false`, `fftHeight`, `maximized=true`, `bandPlanEnabled=false`;
     - `moduleInstances.Radio.enabled = false`.
   - `rtl_sdr_config.json`: `sampleRate`, `gain`, AGCs off.
   - `hackrf_config.json`: `sampleRate`, `lnaGain`, `vgaGain`, amp off, **`bandwidth: 16`**. 16 means Auto; 0 would select a 1.75 MHz baseband filter.
   - `rigctl_server_config.json`: `autoStart: true`, port 4532.
3. `open -a SDR++`, wait for the rigctl port (60 s timeout, 3 attempts; it starts slowly under load).
4. Send `\start` to begin streaming.

### 7.2 Why the Radio module is disabled

Two SDR++ segfaults came from the demodulator VFO (`ImGui::WaterFall::calculateVFOSignalInfo`):
- a VFO parked outside the visible span;
- a RAW-mode VFO wider than a decimated span.

Disabling the Radio module removes the VFO entirely, which also keeps its grey overlay out of the images.

### 7.3 rigctl commands used

| Command | Effect |
|---|---|
| `\start` / `\stop` | Start/stop the source (found with `strings rigctl_server.dylib`) |
| `F <Hz>` / `f` | Set/get frequency |

### 7.4 Screenshots and cropping

- `grab()` runs `screencapture -x -o -l <windowID>`; the window ID comes from `./winlist`.
- `find_waterfall()` locates the colour-mapped region below the FFT plot. The crop box is cached in `calibration.json` per window size: 2747×1191 px on the laptop display, 3435×1393 px on the 4K display.
- The crop excludes the frequency axis, FFT trace and UI, so every image is waterfall only.

---

## 8. Calibration

All values live in `dataset_tools/capture/calibration.json`.

### 8.1 Waterfall scroll speed (how many seconds one screen shows)

**Method:**
1. Transmit an on/off square wave with a known period.
2. Take one screenshot.
3. Count the dash period in pixels.

This needs a single screenshot. An earlier method compared two screenshots one second apart, and it **aliased** whenever the waterfall scrolled more than half a screen in that second. Don't use it.

**Result on this Mac:** 2.0 px per FFT line (Retina), so `t_screen = 1191 / (2 × fft_rate)` s.

**Real-time limit:** SDR++ draws every line only while **FFT size × rate ≤ ~50 M bins/s**:

| Setting | Behaviour |
|---|---|
| 65536 @ 500, 32768 @ 1500, 16384 @ 3000 | Exact, regular scroll |
| 65536 @ 1000 | Borderline |
| 32768 @ 2000+, 65536 @ 1500+ | SDR++ drops lines unevenly; the time axis becomes irregular |

`screen_seconds()` caps the effective rate accordingly.

### 8.2 Frequency error

1. Transmit a CW tone at +20 kHz.
2. Measure its column at 250 kS/s with a 32k FFT.
3. **Result:** +1978 Hz at 433.92 MHz, about 4.6 ppm of combined HackRF/RTL crystal error.

It's corrected by tuning the HackRF LO down by `1978 × f / 433.92 MHz`; the error scales with frequency. Without the correction, narrow zooms (4–8 kHz spans) miss the signal completely.

### 8.3 Brightness: auto-levelling

The goal is the reference SDR++ look: dark-navy noise with visible texture.

**`autolevel()`:**
1. Restart SDR++ at a candidate `min_db` with the TX off.
2. Screenshot and measure `median(B) + 2·median(G)` over the **newest 100 rows only**. At slow frame rates the rest of the waterfall is still black right after a restart, which made the old metric push levels far too bright.
3. Bisect towards the target (70 ± 20):
   - Start near `-84 dB − 10·log10(decimation)`, because decimation lowers the per-bin noise floor; at ÷64 it's about −130 dB.
   - Search exponentially until the target is bracketed.
   - Keep the **best tested** value; an early bug saved an untested value.
4. Cache the result per `(source, rate, decimation, FFT size, gain)`.

`max_db = min_db + range`, with the range randomized from 40 to 60 dB (30 to 45 for HackRF receive).

---

## 9. The bench capture loop

### 9.1 One session (`run_capture.run_class`)

1. **Pick a sub-type and profile** (`dataset_tools/capture/profiles.py`):
   - RTL gain, FFT size and frame rate suited to the signal's timescale (slow modes 30–150 fps; bursty modes up to 3000 fps with small FFTs).
   - **40% of sessions are high-res**: 65536 @ 500–750, 32768 @ 1000–1500, or 16384 @ 2000–3000.
   - About 15% of those deliberately use over-limit settings (32k/65k @ 1–3k fps, "as a user would set it"), flagged `overloaded`.
2. **Pre-generate variant 0 and measure its 99.9%-power bandwidth.**
   - Use 99.9%, not 99%: AM puts 99% of its power in the carrier, so a 99% measure framed AM at 1.3 kHz and cut off the sidebands.
3. **Choose the zoom (`choose_zoom`)**: RTL rate × decimation such that the signal fills 8–60% of the screen.
   - High-res sessions always use a wide 1–2.4 MHz span, where narrow signals become thin lines, like the classic SDR++ screenshot.
   - A 32k–65k FFT in a 30 kHz zoom is a 1–2 s window, which turns into horizontal smear.
4. **Pick a centre** in the ISM band. In the busy 915 band, retune up to 4 times until the TX-off screen is quiet.
5. **Auto-level**, then take a **baseline** TX-off screenshot.
6. **For each of 3 variants:**
   1. Generate a new random variant, regenerating if it wouldn't fit the zoom.
   2. Choose an offset that keeps the whole signal on screen and inside the band, then write the CS8 file.
   3. Start `hackrf_transfer -R`.
   4. **Settle the TX gain**, with up to 4 adjustments:
      - over 45% of signal pixels saturated (orange/red Classic colours) → −8 dB;
      - signal score below baseline + 20 (capped at 80) → +8 dB.
   5. **Grab 2 frames one screen apart.** Each must pass the signal check (`p99.5 − median` of brightness, relative to the baseline); bursty signals are re-grabbed until a burst is on screen.
   6. Save the PNG named by its SHA-256 and append a manifest row.
   7. Stop the TX and delete the CS8.

### 9.2 The balanced driver

`run_full.py` works round-robin:
- each round runs one session (6 images) for every class below target, emptiest first;
- it stops at a time budget (`--hours 1.9`, under the 2 h task limit) and resumes from `manifest.csv` counts;
- stopping at any point leaves the classes balanced.

```bash
cd dataset_tools/capture
caffeinate -dims ../../venv-classidyne/bin/python run_full.py --hours 1.9   # repeat until "all classes at target"
python run_capture.py QPSK --sessions 4      # force a sub-type into its class
```

About 70–180 s per session. Roughly 2,300 bench images took about 12 hours.

### 9.3 Manifest schema (`dataset_tools/manifest.csv`)

| Column | Meaning |
|---|---|
| `file` | `datasets/waterfall/<class>/<sha256>.png` (relative to the repo root) |
| `class` | dataset label |
| `source` | `synthetic` (bench), `real-old` (curated legacy), `real-ota` (live receive-only) |
| `group_id` | capture group: one TX variant, one legacy screenshot session, or one live centre frequency. **Used for leak-free splits.** |
| `session` | capture session (one SDR++ configuration) |
| `rtl_center_hz`, `offset_hz`, `span_hz`, `decimation` | tuning and zoom |
| `fft_size`, `fft_rate`, `rtl_gain`, `tx_gain`, `min_db`, `max_db`, `tx_fs` | SDR++/radio settings |
| `colormap` | SDR++ colormap the frame was rendered in (`Classic`, `Turbo`, ...); empty for legacy images from other tools |
| `params` | JSON: `subtype`, `hires`, `overloaded`, `obw_hz`, plus every generator parameter, or legacy/live details |

### 9.4 Captures in other colormaps

Most images are rendered in SDR++'s **Classic** colormap. Grayscale conversion (which `app.py` applies) does **not** make the colormap irrelevant: most SDR++ maps are not monotonic in brightness. Classic runs dark blue → white → yellow → red → dark red, so the strongest signals turn *dark* in grayscale, while Viridis or Inferno turn them bright.

Two complementary fixes:

1. **Capture in other colormaps.** `run_capture.py --colormap <name>|random` and `run_live.py --colormap random` set SDR++'s `colorMap` for the session and log it in the `colormap` column. The level/signal metrics (auto-level, signal score, saturation) are tuned for Classic, so frames in other maps are first mapped back to signal level and re-rendered in Classic before measuring (`run_capture.as_classic`).

   ```bash
   # ~20 hi-res images per class, a different non-Classic colormap each session
   python dataset_tools/capture/run_full.py --colormap random --hires --target 20 --variants 3 --frames 1
   python dataset_tools/capture/run_live.py wifi-esp32 --frames 20 --colormap random --hires
   ```

2. **Colormap round-trip augmentation** at training time (`dataset_tools/gen/colormaps.py`, `train.py --cmap-aug`). SDR++ interpolates each map's colour stops linearly, so a 256-entry LUT reproduces it exactly. Every SDR++-rendered image is inverted to its colormap index ("signal level") and re-coloured with a random SDR++ map before the usual grayscale step. The LUTs are read from `/Applications/SDR++.app/Contents/Resources/colormaps/` (override with `SDRPP_COLORMAPS`).

---

## 10. Live captures

These are receive-only, in `dataset_tools/capture/run_live.py`; `live_all.sh` runs the suite. Each target entry is `(class, source, centre list, spans, frame-rate range, FFT sizes, ESP32 mode)`.

### 10.1 ESP32 traffic generator (`dataset_tools/esp32_traffic/esp32_traffic.ino`)

Flash it:

```bash
cd dataset_tools/esp32_traffic
arduino-cli compile --fqbn esp32:esp32:esp32:PartitionScheme=min_spiffs .
arduino-cli upload  -p /dev/cu.usbserial-0001 --fqbn esp32:esp32:esp32:PartitionScheme=min_spiffs .
```

It boots **idle**. Serial commands at 115200 baud:

| Command | Effect |
|---|---|
| `w <ch>` | Soft-AP `CLASSIDYNE-TEST` on channel `ch`, plus raw broadcast data frames (40–1440 B) cycling 802.11b DSSS/CCK and 802.11g/n OFDM rates |
| `b` | BLE non-connectable advertising every 20–30 ms |
| `m <ch>` | Wi-Fi + BLE together |
| `d <ms>` | Mean gap between frames |
| `p <q>` | TX power in 0.25 dBm units (default 32 = 8 dBm) |
| `s` | Stop everything |

Notes:
- Raw frames need `esp_wifi_80211_tx(..., en_sys_seq=true)`, or the ESP32 logs warnings.
- The BLE library in core 3.x takes Arduino `String`, not `std::string`.
- Opening the serial port with DTR/RTS low avoids resetting the board.

### 10.2 HackRF receive in SDR++

- **Source:** `"source": "HackRF"`, 4–20 MS/s, LNA 24–40, VGA 16–30, amp off, **baseband filter Auto**.
- **Wi-Fi:** channels 1, 6 and 11, both ESP32 traffic and ambient.
- **Bluetooth: only at 2480 MHz** (BLE advertising channel 39, above US Wi-Fi channel 11), in 4–10 MHz spans.
  - Captures on channels 37/38 and "ambient 2.4 GHz" were dominated by neighbours' Wi-Fi, so they were mixed-label and rejected.
- **Cellular:** US LTE/5G/GSM downlinks: 739/751/763, 875/885, 1940–1980, 2120–2140, 2350, 2630–2660 MHz.
- **ATSC:** UHF channels 14–36 (473–611 MHz centres), 6 MHz 8VSB blocks.
- **HDMI leakage:** pixel-clock harmonics of the attached display, 148.5 MHz (1080p60) and 594 MHz (4K60).
  - Check the bands: 742.5 and 891 MHz landed in LTE downlinks, 1188 MHz in the aviation DME band, so those were rejected.

### 10.3 RTL-SDR off-air

Targets were FM (88.5–107.5 MHz), airband, ADS-B 1090, AIS 162, pagers and VOR.

- **With an indoor antenna only FM broadcast was usable.**
- Airband/VOR "signals" were FM images and spurs.
- AIS, pager and ADS-B had no traffic.
- Re-run these with an outdoor antenna.

### 10.4 Label checks for live data (`dataset_tools/curate/occupancy.py`)

Live frames pass a signal-score check, but a DC spike or one spur can fool it. So:
- **Wideband occupancy:** the fraction of frequency columns more than 12 levels above the noise floor, after a 5-column smoothing so single spurs don't count. Empty frames score about 0.01; real LTE, ATSC and Wi-Fi score 0.08–0.94. Threshold: 0.05.
- **Visual review of every group** with `dataset_tools/eval/contact.py`, which is how the airband/VOR images, the 891/742.5/1188 MHz "HDMI" groups and the mixed bluetooth groups were caught.

Rejected frames move to `tmp/dataset/_live_rejected/<reason>/`. Nothing is deleted.

---

## 11. Curating the legacy dataset

`dataset_tools/curate/curate_existing.py` imports the useful v2 images as `source=real-old`:

1. Drop `unknown`, hidden files and corrupt images.
2. Trim flat UI borders (near-constant edge rows/columns).
3. Apply quality filters: under 200×150 px, aspect ratio beyond 6:1, or nearly uniform (σ < 6).
4. Remove near-duplicates with a 16×16 difference hash (Hamming distance ≤ 12 of 256).
5. Define legacy **sessions** as screenshot-size buckets (`w//40 × h//40`). Cap each session at 40% of the class, so one old capture session can't dominate.
6. Pick a diverse subset by **farthest-point selection** on the hash, up to 60 per bench-synthesized class (150 for real-only classes).

Result: 1,115 of 3,996 legacy images kept. 988 remain after class merges and balancing.

`dataset_tools/curate/merge_classes.py` relabels files and manifest rows for class merges. It keeps the old label as `params.subtype`, re-caps merged legacy images, and writes `known_frequencies.proposed.json`.

---

## 12. Class design and balancing

**Merges** (decided with the dataset owner):

- **`morse` + `remote-keyless-entry` → `OOK`** (the v2 `on-off-keying` class, renamed). All three are on/off keyed carriers. The keyfob 2FSK variant isn't OOK; FSK keyfobs are a sub-kind of the new `2FSK` class.
- **New `2FSK` class** (152 bench images + colormap captures): continuous telemetry, ISM sensor packets (preamble + sync + payload) and FSK keyfob words; 100 b/s–38.4 kb/s, deviation 3–60 kHz, optional Gaussian shaping. Low rates are favoured so the tone switching stays visible on fast, high-resolution waterfalls, and 70% of its sessions use the hi-res look (`HIRES_P` in `profiles.py`), matching a 32k-FFT / 3000 fps SDR++ reference screenshot.
- **`8PSK` + `16QAM` + `32QAM` (+ new BPSK, QPSK) → `psk-qam`.** A waterfall can't separate constellations. RRC-shaped linear modulations occupy `Rs·(1+β)` whatever the constellation. Before the merge, held-out recall was 0.02–0.25 with the classes confused with each other.

**Balancing** (`dataset_tools/curate/balance.py`):
- Cap each class at 150.
- Keep **all** live captures.
- Trim the rest one image at a time from the largest (sub-type, capture group) bucket, so sub-types and sessions stay even. For example, `psk-qam` ends up with 24–32 images of each modulation.
- Move overflow to `tmp/dataset/_balanced_out/`.

**Removed for v3 (planned for v3.5):** drone-video (5), uav-video (6), hdmi (29) and atsc (67). They're too thin to train or test reliably.
- Their images were deleted from v3.
- To bring a class back:
  1. Capture it again: `run_live.py atsc-ota` / `hdmi-leak` for live ATSC and HDMI leakage, an FPV transmitter for the video classes, and the old v2 images via `curate_existing.py`.
  2. Re-run `balance.py`, then `train.py --resplit`.

---

## 13. Training and evaluating RadioNet

All code is in `dataset_tools/train/`.

### 13.1 Splits that measure generalization, not memory

`make_splits()` assigns whole **capture groups** to train, val or test (about 70/15/15 per class). Frames of one transmission, one legacy screenshot session or one live centre frequency are never on both sides. This is the most important difference from the v2 notebooks, whose random per-image splits put near-identical frames in both train and test.

`--extend-split` keeps the assignment of every group already in `splits.csv` and only places new groups. Use it when adding data, so a model trained on the old split can still be compared fairly on the new test set (none of its training images moves into test).

### 13.2 Preprocessing finding

The v2 `app.py` used timm's eval transform: resize the short side to 256, then **centre-crop 224×224**. On a wide waterfall (2747×1191) that throws away the outer 62% of the frequency span. `evaluate.py` measures both:
- `app`: the old centre-crop;
- `full`: the whole image squashed to 224×224 (640 px LANCZOS thumbnail, then bicubic resize). This is what training and the v3 `app.py` use.

### 13.3 Training (`train.py`)

| Setting | Value |
|---|---|
| Starting weights | `--init imagenet` (timm ImageNet weights), `radionet` (v2 checkpoint) or `scratch` |
| Architectures | `--arch efficientnet_b0` (v3, 1280-d) or `resnet34` (512-d) |
| Input | Grayscale → RGB, as in `app.py` |
| Augmentation | RandomResizedCrop (time/frequency crops, scale 0.35–1, ratio 0.4–2.5), brightness/contrast jitter, random gamma; `--cmap-aug 0.5`: half the SDR++-rendered images are re-coloured with a random SDR++ colormap before the grayscale step (§9.4) |
| Excluded augmentations | **No flips or rotations.** A horizontal flip swaps USB/LSB and turns LoRa up-chirps into down-chirps; a vertical flip reverses time |
| Class balance | `WeightedRandomSampler` (1/class count) |
| Loss | Cross-entropy with label smoothing 0.1; optional `--supcon 0.5` (supervised contrastive loss on L2-normalised embeddings, which suits Classidyne's cosine kNN) |
| Optimiser | AdamW, OneCycle LR (backbone 3e-4, head 3e-3), 25 epochs; best checkpoint by validation macro-F1 |
| Outputs | `tmp/dataset/models/RadioNet_<tag>.pth` (backbone + `arch`, `preprocess`, `classes`: the format `app.py` loads) and `*_with_head.pth` |

### 13.4 Evaluation (`evaluate.py`)

For each model × preprocessing, on the held-out test groups:

- **Classidyne kNN:** top-20 cosine vote against the train+val gallery, which is what `/api/classify` does.
- **Classifier head** accuracy and macro-F1, per-class recall, and confusion-matrix PNGs.
- **Domain shift:** models trained on **synthetic only** (`--train-source synthetic`), then tested on real images (legacy and live) with a synthetic gallery.
- **Style-leak baseline:** kNN on size/aspect/mean colour only. Lower is better: it measures how much the label can be guessed without looking at the signal.
- **Colormap shift** (`--cmap-shift`): every SDR++-rendered test image re-rendered in all 14 SDR++ colormaps; plus a row for the real test captures made in non-Classic colormaps.

```bash
cd dataset_tools/train
python train.py --arch efficientnet_b0 --supcon 0.5 --cmap-aug 0.5 --extend-split --tag v3_final
python evaluate.py --v3 ../../tmp/dataset/models/RadioNet_v3_final_with_head.pth --cmap-shift \
                   --out ../../tmp/dataset/reports/report_final.md
```

`dataset_tools/eval/app_eval.py` then evaluates the deployed app end to end (live vector DB, `/api/classify` voting rule, HTTP latency); see [APP_EVALUATION.md](APP_EVALUATION.md).

Results of the v3 runs (2026-10-03, 24 classes; full tables in `tmp/dataset/reports/`, summary in [DATASET_V3.md](DATASET_V3.md)).

Model selection, first 24-class round (571 held-out images, Classidyne kNN macro-F1, whole frame):

| Model | Classic test images | Mean over all 14 SDR++ colormaps |
|---|---|---|
| RadioNet v2 (ResNet-34) | 0.58 | 0.40 |
| EfficientNet-B0 + SupCon | 0.84 | 0.71 |
| EfficientNet-B0 + SupCon + colormap augmentation | 0.85 | 0.86 |

Final (630 held-out images, after the colormap captures):

| Model | kNN macro-F1 | Real captures in other colormaps | Colormap mean |
|---|---|---|---|
| RadioNet v2 | 0.57 | 0.28 | 0.37 |
| EfficientNet-B0 + SupCon + cmap-aug, 224×224 | 0.83 | 0.62 | 0.81 |
| **same, 448×224 input (deployed)** | **0.87** | **0.64** | **0.84** |

Earlier 27-class experiments (2026-10-03, same protocol) ranked the alternatives: ResNet-34 + SupCon (ImageNet start) 0.80, ResNet-34 from the v2 RadioNet weights 0.78, EfficientNet-B0 without SupCon 0.79, ResNet-34 from scratch 0.47.

Lessons:
- **ImageNet start beats training from scratch** at this dataset size.
- **Whole-image preprocessing beats the centre-crop** for every model.
- **Wide input for wide waterfalls.** At 224 px, narrow signals in a 1–2.4 MHz span shrink to a few pixels: high-res frames with < 3% occupancy were 37% correct vs 93% for the rest. 448×224 lifted 2FSK from 0.31 to 0.73.
- **Colormap augmentation is nearly free robustness**; real captures in other colormaps add more.
- **Bench-only training transfers poorly to other tools' screenshots** (style gap), so keep mixing in real captures.

### 13.5 Using a new model in Classidyne

1. Copy `tmp/dataset/models/RadioNet_<tag>.pth` over `RadioNet/RadioNet.pth` (Git LFS). The checkpoint stores `arch`, `preprocess` and `input_size`; `app.RadioNetExtractor` reads them, so no code change is needed. (`CLASSIDYNE_MODEL=<path>` tries a checkpoint without copying.)
2. Delete `classidyne_db/` and re-embed (`POST /api/start-embedding`): embeddings from different models are not comparable.
3. Run `python -m pytest tests/` and `python dataset_tools/eval/app_eval.py` against the running server.

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `could not create image from window` | No Screen Recording permission | System Settings → Privacy & Security → Screen & System Audio Recording |
| Mouse clicks do nothing (`AXIsProcessTrusted() == false`) | Accessibility not granted to the process | Not needed: use rigctl `\start` and config edits |
| SDR++ segfault in `calculateVFOSignalInfo` | VFO outside the span, or RAW VFO wider than the decimated span | Disable the Radio module |
| `rigctl server did not come up` | SDR++ starts slowly while training uses the GPU | 60 s timeout + 3 retries |
| Narrow-zoom captures are empty | ~2 kHz crystal error at 434 MHz | Frequency-error correction (§8.2) |
| Waterfall saturated yellow/orange | `min_db` too low; or fresh, half-empty waterfall measured | Auto-level on the newest rows only |
| Auto-level never converges at ÷32/÷64 | Noise floor near −130 dB; search started at −84 dB with 4 dB steps | Decimation-aware start + exponential search |
| Horizontal splatter lines | Hard keying / phase jumps | Raised-cosine ramps, `cumsum` FSK |
| Python killed (exit 137) | Huge FFT convolution | `oaconvolve` |
| Wide signals hang off the edge | Fixed per-class zoom | Zoom chosen from the measured 99.9% bandwidth |
| AM sidebands cut off | 99% bandwidth ≈ carrier only | 99.9% bandwidth |
| ADS-B/DAB "don't fit the ISM band" | 2.3 MHz PPM > 1.74 MHz band | Band-limit ADS-B to 1.5 MHz; >1.6 MHz signals → 902–928 MHz |
| 915 MHz frames show other devices | Busy band | Busy-baseline retune; prefer 433 MHz |
| Fast-rate screenshots overlap | Scroll measured with two aliased screenshots; big FFTs over the real-time limit | Single-shot square-wave method; cap effective rate at 50 M bins/s |
| HackRF RX shows only ~3 MHz of a 20 MHz span | `bandwidth: 0` = 1.75 MHz filter | `bandwidth: 16` (Auto) |
| Orphan `hackrf_transfer` after a kill | `-R` loops forever | atexit/signal handlers + `pkill -INT -x hackrf_transfer` |
| `rm -rf` of a dataset folder blocked | Safety check on glob deletes | Move to an `_archive` folder instead |
| zsh `$P` "command not found" | zsh doesn't word-split variables | Use `PY=...; $PY script.py` with a single word |
| Live Wi-Fi in hi-res mode barely visible | A 2–4 ms FFT window dilutes sub-ms bursts; with a 30–45 dB range they only reach the bottom of the colormap | Hi-res HackRF captures use a 20–30 dB range |
| Captures in GQRX / Inferno look almost black | Those maps start at black; weak signals stay dark | Expected look; frames whose re-rendered signal score is < 25 are archived (`tmp/dataset/_faint_colormap/`) |
| `/api/classify` took ~1.5 s | Collage decoded 20 full-resolution PNGs serially | Parallel decode (`TILE_POOL`): ~0.36 s |
| Port 5000 busy on macOS | AirPlay Receiver | The app picks 5001; set `CLASSIDYNE_PORT=5001` for tests |

---

## 15. File map

```
dataset_tools/                    (tracked)
  README.md                       quick reference
  paths.py                        shared locations (repo root, manifest, splits, scratch)
  manifest.csv                    one row per image (schema §9.3)
  splits.csv                      reference group-held-out train/val/test split
  gen/        signals.py          all generators (+ CLASS_OF / SUBTYPES)
              iqlib.py            DSP helpers (from aft-rfctf)
              colormaps.py        SDR++ colormap LUTs, inverse map, re-colouring
              preview.py          offline spectrogram check
  capture/    sdrpp_ctl.py        SDR++ config/restart/rigctl/screenshot control
              run_capture.py      bench capture loop, TX safety, calibration, auto-level
              run_full.py         resumable balanced round-robin driver (+ colormap top-ups)
              profiles.py         per-class display profiles + hi-res mode
              run_live.py         receive-only live targets (HackRF / RTL / ESP32)
              live_all.sh         live suite
              winlist.swift       window id helper (CoreGraphics; build to capture/winlist)
              calibration.json    crop box, px/line, frequency error, level cache (machine-specific)
  esp32_traffic/esp32_traffic.ino Wi-Fi/BLE traffic generator firmware
  curate/     curate_existing.py  legacy import (dedup, borders, quality, caps)
              merge_classes.py    class merges
              occupancy.py        wideband label check for live frames
              balance.py          cap classes at 150, diversity-preserving
  train/      common.py           splits, caches, model, preprocessing (app vs full)
              train.py            training
              evaluate.py         model comparison report + confusion PNGs
  eval/       app_eval.py         end-to-end evaluation of the running app
              contact.py          contact sheets for visual review
datasets/waterfall/<class>/       the dataset (PNG, sha256 names; git-ignored)
tmp/dataset/                      scratch (git-ignored): iq/, models/, reports/, logs/, train_cache/,
                                  _*/ archived frames (rejected, overflow, pilots) - nothing deleted
tmp/backups/                      SDR++ config, v2 waterfall dataset, v2 RadioNet.pth, v2 vector DB
```

---

## 16. Merging and publishing

The v3 merge was done like this (repeat it for a future version):

1. Review the classes visually: `python dataset_tools/eval/contact.py out.png <class> --n 12`.
2. Move the old waterfall folder aside (`tmp/backups/datasets_v2/waterfall`) and put the new one at `datasets/waterfall`. `datasets/fft` is untouched.
3. Update `known_frequencies.json` for renamed / merged / new classes (`merge_classes.py` writes a proposal).
4. Train, then copy `tmp/dataset/models/RadioNet_<tag>.pth` over `RadioNet/RadioNet.pth` (Git LFS). The checkpoint carries `arch` and `preprocess`, so `app.py` needs no code change for EfficientNet-B0.
5. `rm -rf classidyne_db` (or move it aside), start `python app.py`, `POST /api/start-embedding`.
6. Run `python -m pytest tests/` and `python dataset_tools/eval/app_eval.py` against the running server.
7. For Kaggle, the screenshots are about 3 MB each (~12 GB total). Either publish the full-resolution PNGs or add a downscaled copy (e.g. 1024 px wide, still far above the 224 px model input). Include `manifest.csv` and `splits.csv` so others use **group-held-out** splits.
8. Restore your SDR++ settings from `tmp/backups/sdrpp/` (quit SDR++ first).
