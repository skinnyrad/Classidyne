"""Drive SDR++ without the GUI.

* Display/source settings are only read at startup, so a profile change is:
  quit SDR++ -> edit config.json / rtl_sdr_config.json / hackrf_config.json -> relaunch.
* Runtime control goes through the rigctl server plugin (autoStart enabled):
  ``\\start`` / ``\\stop`` the source and ``F <hz>`` to retune.
* Screenshots: ``screencapture -l <windowID>`` (window ID from ./winlist), cropped to the waterfall.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

CFG_DIR = Path.home() / "Library/Application Support/sdrpp"
HERE = Path(__file__).resolve().parent
APP = "SDR++"
RIGCTL = ("localhost", 4532)


# ----------------------------------------------------------------------------- config

def _edit(name: str, fn) -> None:
    p = CFG_DIR / name
    cfg = json.loads(p.read_text())
    fn(cfg)
    p.write_text(json.dumps(cfg, indent=4))


def running() -> bool:
    return subprocess.run(["pgrep", "-x", "sdrpp"], capture_output=True).returncode == 0


def quit_app(timeout=15) -> None:
    if not running():
        return
    subprocess.run(["osascript", "-e", f'quit app "{APP}"'], capture_output=True)
    t0 = time.time()
    while running() and time.time() - t0 < timeout:
        time.sleep(0.3)
    if running():
        subprocess.run(["pkill", "-TERM", "-x", "sdrpp"])
        time.sleep(2)


def apply_profile(p: dict) -> None:
    """p keys: source ('RTL-SDR'|'HackRF'), sample_rate, gain, freq, fft_size, fft_rate,
    min_db, max_db, colormap. SDR++ must not be running (it rewrites config on exit)."""
    assert not running(), "quit SDR++ before editing its config"

    def main(c):
        c.update({
            "source": p.get("source", "RTL-SDR"),
            "frequency": float(p["freq"]),
            "fftSize": int(p.get("fft_size", 32768)),
            "fftRate": int(p.get("fft_rate", 60)),
            "colorMap": p.get("colormap", "Classic"),
            "min": float(p.get("min_db", -100)),
            "max": float(p.get("max_db", -20)),
            "showMenu": False,
            "showWaterfall": True,
            "fftHeight": int(p.get("fft_height", 250)),
            "fftHold": False,
            "fftSmoothing": False,
            "fullWaterfallUpdate": False,
            "centerTuning": False,
            "bandPlanEnabled": False,
            "decimation": int(p.get("decimation", 1)),
            "maximized": True,
        })
        # No demod VFO: with decimation the RAW VFO is wider than the span and SDR++ segfaults
        # (ImGui::WaterFall::calculateVFOSignalInfo). It is also not wanted in the images.
        c["moduleInstances"]["Radio"]["enabled"] = False
        c["vfoOffsets"] = {}
    _edit("config.json", main)

    if p.get("source", "RTL-SDR") == "RTL-SDR":
        def rtl(c):
            d = c["devices"][c["device"]]
            d.update({"sampleRate": float(p["sample_rate"]), "gain": int(p.get("gain", 15)),
                      "rtlAgc": False, "tunerAgc": False, "biasT": False, "offsetTuning": False})
        _edit("rtl_sdr_config.json", rtl)
    else:
        def hrf(c):
            serial = c.get("device") or next((k for k in c.get("devices", {}) if k), None)
            if serial:
                c["device"] = serial
                d = c["devices"].setdefault(serial, {})
                d.update({"sampleRate": float(p["sample_rate"]), "lnaGain": int(p.get("lna", 24)),
                          "vgaGain": int(p.get("vga", 20)), "amp": False, "biasT": False,
                          "bandwidth": 16})  # 16 = Auto (index 0 would be a 1.75 MHz filter)
        _edit("hackrf_config.json", hrf)

    def rig(c):
        c["Rigctl Server"].update({"autoStart": True, "host": "localhost", "port": RIGCTL[1]})
    _edit("rigctl_server_config.json", rig)


def launch(timeout=30) -> None:
    subprocess.run(["open", "-a", APP])
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(RIGCTL, timeout=1):
                time.sleep(1.5)
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("SDR++ rigctl server did not come up")


# ----------------------------------------------------------------------------- rigctl

def rig(cmd: str) -> str:
    with socket.create_connection(RIGCTL, timeout=3) as s:
        s.sendall((cmd + "\n").encode())
        time.sleep(0.2)
        try:
            return s.recv(4096).decode(errors="replace").strip()
        except socket.timeout:
            return ""


def start():
    return rig("\\start")


def stop():
    return rig("\\stop")


def tune(hz: float):
    return rig(f"F {int(hz)}")


def restart_with(profile: dict, attempts=3) -> None:
    for i in range(attempts):
        quit_app()
        apply_profile(profile)
        try:
            launch(timeout=60)
            start()
            return
        except RuntimeError:
            if i == attempts - 1:
                raise
            time.sleep(5)  # slow start under load: quit and try again


# ----------------------------------------------------------------------------- screenshots

def window() -> tuple[int, int, int, int, int]:
    out = subprocess.run([str(HERE / "winlist"), APP], capture_output=True, text=True, check=True).stdout
    wid, x, y, w, h = map(int, out.split())
    return wid, x, y, w, h


def grab(path: Path) -> Image.Image:
    wid = window()[0]
    subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), str(path)], check=True)
    return Image.open(path).convert("RGB")


def find_waterfall(img: Image.Image) -> tuple[int, int, int, int]:
    """Locate the waterfall: the large region below the FFT plot whose pixels are
    colormap-coloured (not the UI greys). Returns (left, top, right, bottom) in px."""
    a = np.asarray(img).astype(np.int16)
    sat = (a.max(2) - a.min(2)) > 25            # coloured (waterfall) vs grey UI
    rows = sat.mean(1)
    cols = sat.mean(0)
    ys = np.where(rows > 0.6)[0]
    xs = np.where(cols > 0.5)[0]
    # waterfall is the longest contiguous run of coloured rows/cols
    def longest(idx):
        best, cur = (0, 0), [idx[0], idx[0]]
        for v in idx[1:]:
            if v == cur[1] + 1:
                cur[1] = v
            else:
                if cur[1] - cur[0] > best[1] - best[0]:
                    best = tuple(cur)
                cur = [v, v]
        return max(best, tuple(cur), key=lambda r: r[1] - r[0])
    y0, y1 = longest(ys)
    x0, x1 = longest(xs)
    m = 6  # stay clear of the borders
    return x0 + m, y0 + m, x1 - m, y1 - m
