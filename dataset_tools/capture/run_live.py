"""Receive-only captures of real signals (source=real-ota) in SDR++.

* HackRF source (10-20 MS/s): wifi + bluetooth (ESP32 traffic generator and ambient), live cellular downlinks.
* RTL-SDR source: off-air FM broadcast, airband, ADS-B, AIS, pagers, NOAA/VOR etc.
Nothing is transmitted by this script except the ESP32's low-power 2.4 GHz traffic.

usage: python run_live.py <target> [...] [--frames 40] [--seed 1]
       python run_live.py --scan cellular      # survey candidate centres, print signal scores
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import run_capture as rc
import sdrpp_ctl as sdr

ESP_PORT = "/dev/cu.usbserial-0001"

# target -> class, source, candidate centres (Hz), spans (sample rates), fft_rate range, fft sizes, esp32 mode
TARGETS = {
    "wifi-esp32":      ("wifi", "HackRF", [2412e6, 2437e6, 2462e6], [20e6, 16e6], (300, 2500), [2048, 4096, 8192], "w"),
    "wifi-ambient":    ("wifi", "HackRF", [2412e6, 2437e6, 2462e6, 5180e6, 5745e6], [20e6], (200, 2000), [2048, 4096, 8192], None),
    # BLE advertising channel 39 (2480 MHz) sits above US Wi-Fi ch 11, so captures there are BLE-only.
    # (ch 37/38 and "ambient" 2.4 GHz captures were dominated by neighbouring Wi-Fi: mixed labels, dropped.)
    "bluetooth-esp32": ("bluetooth", "HackRF", [2479.5e6, 2480e6, 2480.5e6, 2481e6], [4e6, 8e6, 10e6],
                        (500, 3000), [1024, 2048, 4096], "b"),
    # US LTE/5G/GSM downlink bands (receive only)
    "cellular":        ("cellular", "HackRF", [739e6, 751e6, 763e6, 875e6, 885e6, 1940e6, 1960e6, 1980e6,
                                               2120e6, 2140e6, 2350e6, 2630e6, 2660e6], [20e6, 10e6],
                        (100, 1200), [4096, 8192, 16384], None),
    "fm-ota":          ("fm", "RTL-SDR", [float(f) * 1e6 for f in np.arange(88.5, 108, 1.0)], [2.4e6, 1.024e6],
                        (40, 300), [16384, 32768], None),
    "airband-ota":     ("airband", "RTL-SDR", [118.5e6, 119.5e6, 120.5e6, 121.5e6, 124.5e6, 127.5e6, 132.5e6, 135.5e6],
                        [2.4e6, 1.024e6], (30, 120), [16384, 32768], None),
    "ads-b-ota":       ("ads-b", "RTL-SDR", [1090e6], [2.4e6, 2.048e6], (1500, 3000), [2048, 4096], None),
    "ais-ota":         ("ais", "RTL-SDR", [162.0e6], [250e3], (150, 500), [8192, 16384], None),
    "pocsag-ota":      ("pocsag", "RTL-SDR", [152.0e6, 157.5e6, 929.5e6, 931.5e6], [1.024e6, 2.4e6], (100, 300),
                        [16384, 32768], None),
    # US ATSC 1.0 UHF channels 14-36 (6 MHz, 8VSB with pilot); receive only
    "atsc-ota":        ("atsc", "HackRF", [float(473 + 6 * k) * 1e6 for k in range(23)], [8e6, 10e6, 20e6],
                        (60, 400), [8192, 16384, 32768], None),
    # HDMI/TMDS leakage from the attached display cable: pixel-clock harmonics (1080p60 148.5 MHz, 4K60 594 MHz)
    "hdmi-leak":       ("hdmi", "HackRF", [148.5e6, 297e6, 445.5e6, 594e6, 742.5e6, 891e6, 1188e6], [10e6, 20e6],
                        (30, 200), [16384, 32768, 65536], None),
    "vor-ota":         ("vor", "RTL-SDR", [109e6, 111e6, 113e6, 115e6, 117e6], [2.4e6], (40, 150), [16384, 32768], None),
}


def esp(cmd: str | None):
    if cmd is None:
        return
    import serial
    with serial.Serial(ESP_PORT, 115200, timeout=1) as s:
        s.dtr = False
        s.rts = False
        time.sleep(0.2)
        s.write((cmd + "\n").encode())
        time.sleep(0.5)
        print("esp32:", s.read(400).decode(errors="replace").strip().splitlines()[-1:])


def profile(source, center, sr, rng, fft_rate, fft_sizes):
    p = dict(source=source, freq=center, sample_rate=int(sr), fft_size=int(rng.choice(fft_sizes)),
             fft_rate=int(rng.integers(*fft_rate)), decimation=1, range_db=float(rng.uniform(40, 60)))
    if source == "HackRF":
        p.update(lna=int(rng.choice([24, 32, 40])), vga=int(rng.choice([16, 24, 30])), gain=None, min_db=-90,
                 range_db=float(rng.uniform(30, 45)))  # narrower range = more contrast for weak 2.4 GHz bursts
    else:
        p.update(gain=int(rng.choice([25, 30, 35, 40])), min_db=-85)
    return p


def esp_cmd(mode, center):
    if mode == "w":
        ch = int(round((center - 2407e6) / 5e6))
        return f"w {min(max(ch, 1), 13)}"
    return mode


def run(target: str, frames: int, seed: int, min_score: float, per_center: int, colormap: str | None = None,
        hires: bool = False):
    cls, source, centers, spans, fft_rate, fft_sizes, mode = TARGETS[target]
    rng = np.random.default_rng(seed)
    got, passes = 0, 1
    order = list(rng.permutation(len(centers)))
    try:
        while got < frames and order:
            center = centers[order.pop(0)]
            for _ in range(per_center):
                if got >= frames:
                    break
                rc.CMAP["name"] = (str(rng.choice([n for n in rc.colormaps.luts() if n != "Classic"]))
                                   if colormap == "random" else colormap or "Classic")
                p = profile(source, center, float(rng.choice(spans)), rng, fft_rate, fft_sizes)
                if hires:  # big FFT + fast waterfall, as in run_capture's hi-res sessions
                    p["fft_size"], p["fft_rate"] = [(65536, 600), (32768, 1200), (16384, 2500)][int(rng.integers(3))]
                    if source == "HackRF":  # a long FFT window dilutes short bursts: tighter range keeps them bright
                        p["range_db"] = float(rng.uniform(20, 30))
                p["colormap"] = rc.CMAP["name"]
                p = rc.autolevel(p)
                esp(esp_cmd(mode, center) if mode else None)
                if mode == "w":
                    esp(f"d {int(rng.choice([1, 2, 4, 8, 16]))}")
                time.sleep(1)
                probe = rc.grab_crop("probe")
                t_screen = rc.screen_seconds(p["fft_rate"], probe.height, p["fft_size"])
                time.sleep(t_screen + 0.5)
                for k in range(3):
                    img, score = rc.grab_with_signal(f"{target}-{got}", t_screen, min_score=min_score, tries=4)
                    if score < min_score:
                        print(f"[{target}] {center/1e6:.1f} MHz: weak (score {score:.0f}), next centre", flush=True)
                        break
                    rel, h = rc.save_png(img, cls)
                    rc.log({"file": rel, "class": cls, "source": "real-ota", "group_id": f"{target}-{int(center)}",
                            "session": f"{target}-{seed}", "rtl_center_hz": int(center), "offset_hz": 0,
                            "span_hz": int(p["sample_rate"]), "decimation": 1, "fft_size": p["fft_size"],
                            "fft_rate": p["fft_rate"], "rtl_gain": p.get("gain") or p.get("lna"), "tx_gain": "",
                            "min_db": p["min_db"], "max_db": p["max_db"], "tx_fs": "", "colormap": rc.CMAP["name"],
                            "params": json.dumps({"device": source, "esp32": mode, "vga": p.get("vga")})})
                    got += 1
                    print(f"[{target}] {center/1e6:.1f} MHz span={p['sample_rate']/1e6:.1f}M "
                          f"rate={p['fft_rate']} score={score:.0f} ({got}/{frames})", flush=True)
                    time.sleep(t_screen)
                else:
                    continue
                break
            if not order and got < frames and passes < 2:
                order = list(rng.permutation(len(centers)))  # one more pass with fresh settings
                passes += 1
    finally:
        esp("s" if mode else None)
        rc.CMAP["name"] = "Classic"
    print(f"[{target}] done: {got}/{frames} frames", flush=True)


def scan(target: str):
    cls, source, centers, spans, fft_rate, fft_sizes, mode = TARGETS[target]
    rng = np.random.default_rng(0)
    for c in centers:
        p = rc.autolevel(profile(source, c, spans[0], rng, fft_rate, fft_sizes))
        esp(esp_cmd(mode, c) if mode else None)
        time.sleep(rc.screen_seconds(p["fft_rate"], 1191) + 1)
        print(f"{target} {c/1e6:9.3f} MHz score={rc.signal_score(rc.grab_crop('scan')):.0f}", flush=True)
    esp("s" if mode else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-score", type=float, default=30)
    ap.add_argument("--per-center", type=int, default=2)
    ap.add_argument("--scan")
    ap.add_argument("--colormap", help='SDR++ colormap name, or "random" (any but Classic, per centre)')
    ap.add_argument("--hires", action="store_true")
    a = ap.parse_args()
    if a.scan:
        scan(a.scan)
    for t in a.targets:
        run(t, a.frames, a.seed, a.min_score, a.per_center, colormap=a.colormap, hires=a.hires)
