"""Bench capture: NumPy signal -> HackRF TX -> RTL-SDR -> SDR++ waterfall screenshot.

usage:
  python run_capture.py <class> [<class> ...] --sessions 25 --variants 3 --frames 2
  python run_capture.py --calibrate          # measure waterfall scroll speed (px/s per fft_rate)

RF safety: everything is transmitted inside 433.05-434.79 MHz ISM, HackRF amp off, TX VGA <= 30 dB.
The HackRF LO is parked outside the RTL span so its DC/LO leakage never appears in an image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "gen")]
from paths import DATA, MANIFEST, ROOT, SCRATCH  # noqa: E402
import sdrpp_ctl as sdr  # noqa: E402
from signals import CLASS_OF, GENERATORS, SUBTYPES  # noqa: E402
import colormaps  # noqa: E402
from profiles import PROFILES, pick  # noqa: E402

OUT = DATA
IQ = SCRATCH / "iq"
SHOTS = SCRATCH / "iq" / "_shots"
CALIB = HERE / "calibration.json"

ISM_BANDS = [(433.05e6, 434.79e6), (902.0e6, 928.0e6)]  # only ever transmit inside these


def band_of(hz: float) -> tuple[float, float]:
    for lo, hi in ISM_BANDS:
        if lo <= hz <= hi:
            return lo, hi
    raise ValueError(f"{hz / 1e6:.3f} MHz is outside the ISM bands")
FIELDS = ["file", "class", "source", "group_id", "session", "rtl_center_hz", "offset_hz", "span_hz",
          "decimation", "fft_size", "fft_rate", "rtl_gain", "tx_gain", "min_db", "max_db", "tx_fs", "colormap", "params"]

# SDR++ colormap of the running profile. The level/signal metrics below are tuned for Classic, so frames in
# other colormaps are mapped back to signal level and re-rendered in Classic before measuring.
CMAP = {"name": "Classic"}


def as_classic(img: Image.Image) -> Image.Image:
    if CMAP["name"] == "Classic":
        return img
    return colormaps.recolor(colormaps.to_level(img, CMAP["name"]), "Classic")


# ----------------------------------------------------------------------------- TX file

def write_cs8(iq: np.ndarray, gen_fs: float, tx_fs: float, shift_hz: float, path: Path, peak=100) -> None:
    fr = Fraction(int(tx_fs), int(gen_fs)).limit_denominator(2000)
    y = resample_poly(iq.astype(np.complex64), fr.numerator, fr.denominator).astype(np.complex64)
    n = np.arange(len(y), dtype=np.float64)
    y *= np.exp(2j * np.pi * shift_hz * n / tx_fs).astype(np.complex64)
    y *= peak / (np.max(np.abs(y)) + 1e-9)
    out = np.empty(2 * len(y), np.int8)
    out[0::2] = np.clip(np.round(y.real), -127, 127)
    out[1::2] = np.clip(np.round(y.imag), -127, 127)
    out.tofile(path)


def tx_plan(span: float, offset: float):
    """Choose HackRF sample rate + LO so the LO sits outside the visible span."""
    if span <= 300e3:
        tx_fs, lo = 2e6, span / 2 + 250e3
    elif span <= 1.1e6:
        tx_fs, lo = 4e6, span / 2 + 400e3
    else:
        tx_fs, lo = 8e6, span / 2 + 1.0e6
    return tx_fs, lo, offset - lo  # signal shift relative to the HackRF LO


class Transmitter:
    """hackrf_transfer -R wrapper; always stops the radio on exit or SIGTERM/SIGINT."""
    _live: list = []

    def __init__(self):
        self.p = None
        Transmitter._live.append(self)

    def start(self, path: Path, lo_hz: float, tx_fs: float, gain: int):
        assert any(lo - 5e6 < lo_hz < hi + 5e6 for lo, hi in ISM_BANDS)
        self.stop()
        self.p = subprocess.Popen(["hackrf_transfer", "-t", str(path), "-f", str(int(lo_hz)), "-s", str(int(tx_fs)),
                                   "-x", str(int(min(gain, 30))), "-a", "0", "-R"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        if self.p.poll() is not None:
            raise RuntimeError("hackrf_transfer exited early")

    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.send_signal(2)
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.p = None


# ----------------------------------------------------------------------------- screen

def screen_seconds(fft_rate: int, height_px: int, fft_size: int = 0) -> float:
    """Seconds of signal one screen shows. Above ~50M FFT bins/s SDR++ cannot keep up, so the effective
    line rate is capped (measured with the square-wave scroll test)."""
    c = json.loads(CALIB.read_text()) if CALIB.exists() else {"px_per_fft_line": 1.0}
    eff = min(fft_rate, 50e6 / fft_size) if fft_size else fft_rate
    return height_px / (c["px_per_fft_line"] * eff)


def grab_crop(tag: str) -> Image.Image:
    SHOTS.mkdir(parents=True, exist_ok=True)
    raw = SHOTS / f"{tag}.png"
    img = sdr.grab(raw)
    raw.unlink(missing_ok=True)
    cal = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    box = tuple(cal["box"]) if cal.get("box") and tuple(cal.get("window_px", ())) == img.size else None
    if box is None:
        box = tuple(int(v) for v in sdr.find_waterfall(img))
        if CALIB.exists():
            cal.update(box=box, window_px=img.size)
            CALIB.write_text(json.dumps(cal, indent=2))
    return img.crop(box)


def save_png(img: Image.Image, cls: str) -> tuple[str, str]:
    d = OUT / cls
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "_tmp.png"
    img.save(tmp, optimize=True)
    h = hashlib.sha256(tmp.read_bytes()).hexdigest()
    dst = d / f"{h}.png"
    tmp.rename(dst)
    return str(dst.relative_to(ROOT)), h


def log(row: dict):
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------- main loops

def cal() -> dict:
    return json.loads(CALIB.read_text()) if CALIB.exists() else {}


def noise_metric(img: Image.Image) -> float:
    """Brightness of the noise floor in the Classic colormap (B + 2G of the median pixel)."""
    img = as_classic(img)
    a = np.asarray(img.resize((img.width // 8, img.height // 8), Image.BOX), np.float32)  # smooth speckle
    return float(np.median(a[..., 2]) + 2 * np.median(a[..., 1]))


def autolevel(prof_sdr: dict, target=70.0, tol=20.0, max_iter=8) -> dict:
    """Restart SDR++ until the noise floor renders as dark navy (like the reference SDR++ shots).
    Results are cached per (rate, decimation, fft size, gain)."""
    c = cal()
    key = f"{prof_sdr.get('source', 'RTL-SDR')}/{prof_sdr['sample_rate']}/{prof_sdr['decimation']}/" \
          f"{prof_sdr['fft_size']}/{prof_sdr.get('gain', prof_sdr.get('lna'))}"
    levels = c.setdefault("levels", {})
    # decimation lowers the per-bin noise floor (~10*log10(dec)); start the search near it
    guess = prof_sdr["min_db"] - 10 * np.log10(max(1, prof_sdr.get("decimation", 1)))
    mn = levels.get(key, guess)
    lo_db, hi_db = None, None  # bracket: lo_db too bright (metric high), hi_db too dark
    tried = []
    step = 6.0
    for _ in range(max_iter):
        prof_sdr.update(min_db=round(mn, 1), max_db=round(mn + prof_sdr["range_db"], 1))
        sdr.restart_with(prof_sdr)
        time.sleep(2.5)
        img = grab_crop("level")
        m = noise_metric(img.crop((0, 0, img.width, min(img.height, 100))))  # newest rows only (top)
        tried.append((abs(m - target), mn))
        if abs(m - target) <= tol:
            break
        if m > target:
            lo_db = mn
            mn = (mn + hi_db) / 2 if hi_db is not None else mn + step
        else:
            hi_db = mn
            mn = (mn + lo_db) / 2 if lo_db is not None else mn - step
        if lo_db is None or hi_db is None:
            step *= 2  # exponential search until the target is bracketed
    best_err, mn = min(tried)
    if mn != prof_sdr["min_db"]:  # last restart was not at the best tested level
        prof_sdr.update(min_db=round(mn, 1), max_db=round(mn + prof_sdr["range_db"], 1))
        sdr.restart_with(prof_sdr)
        time.sleep(2.5)
    print(f"  autolevel {key}: min_db={mn:.1f} (err {best_err:.0f}, {len(tried)} tries)", flush=True)
    levels[key] = round(mn, 1)
    c2 = cal()
    c2.setdefault("levels", {}).update(levels)
    CALIB.write_text(json.dumps(c2, indent=2))
    return prof_sdr


def signal_score(img: Image.Image) -> float:
    """How much brighter than the noise floor the hottest 0.5% of (smoothed) pixels are."""
    img = as_classic(img)
    a = np.asarray(img.convert("L").resize((img.width // 4, img.height // 4), Image.BOX), np.float32)
    return float(np.percentile(a, 99.5) - np.median(a))


def grab_with_signal(tag: str, t_screen: float, min_score=35.0, tries=8) -> tuple[Image.Image, float]:
    """Bursty signals can leave a whole screen empty: re-grab (waiting a fraction of a screen) until visible."""
    best, best_s = None, -1.0
    for i in range(tries):
        img = grab_crop(tag)
        sc = signal_score(img)
        if sc > best_s:
            best, best_s = img, sc
        if sc >= min_score:
            break
        time.sleep(max(0.3, 0.5 * t_screen))
    return best, best_s


def hot_fraction(img: Image.Image) -> float:
    """Share of the *signal* pixels that sit at the top of the Classic colormap (orange/red = saturated)."""
    img = as_classic(img)
    small = img.resize((img.width // 4, img.height // 4), Image.BOX)
    a = np.asarray(small, np.int16)
    lum = np.asarray(small.convert("L"), np.float32)
    sig = lum > np.median(lum) + 40
    hot = (a[..., 0] > 200) & (a[..., 2] < 90)
    return float(hot.sum() / max(sig.sum(), 1)) if sig.sum() > 0.002 * sig.size else 0.0


def occupied_bw(iq: np.ndarray, fs: float, frac=0.999) -> float:
    """Bandwidth holding ``frac`` of the power. 99.9% (not 99%) so AM/VOR sidebands under a strong
    carrier still count, while the slow -40 dB skirts of hard-keyed FSK do not."""
    n = 4096
    segs = iq[: (len(iq) // n) * n].reshape(-1, n)[:: max(1, len(iq) // n // 200)]
    p = np.fft.fftshift((np.abs(np.fft.fft(segs * np.hanning(n), axis=1)) ** 2).mean(0))
    c = np.cumsum(p) / p.sum()
    lo, hi = np.searchsorted(c, (1 - frac) / 2), np.searchsorted(c, 1 - (1 - frac) / 2)
    return float(max(hi - lo, 1) * fs / n)


def settle_gain(tx, path, lo_hz, tx_fs, gain, t_screen, need=30.0, lo_g=0, hi_g=30):
    """Adjust HackRF TX gain until the signal is visible (score >= need, i.e. above the TX-off baseline)
    but not a saturated block."""
    for _ in range(4):
        time.sleep(min(t_screen, 3.0) + 0.5)
        img, sc = grab_with_signal("gain", t_screen, min_score=need, tries=3)
        hot = hot_fraction(img)
        if hot > 0.45 and gain > lo_g:
            gain = max(lo_g, gain - 8)
        elif sc < need and gain < hi_g:
            gain = min(hi_g, gain + 8)
        else:
            break
        tx.start(path, lo_hz, tx_fs, gain)
    return gain


RTL_RATES = [250_000, 1_024_000, 2_048_000, 2_400_000]
DECIMATIONS = [1, 2, 4, 8, 16, 32, 64]


def choose_zoom(sub: str, obw: float, rng, prof) -> tuple[int, int]:
    """Pick (RTL sample rate, decimation) so the signal occupies a sensible share of the screen."""
    pref_sr, pref_dec = PROFILES[sub][0], PROFILES[sub][1]
    combos = [(sr, d) for sr in RTL_RATES for d in DECIMATIONS if sr / d >= 3_000]
    def fill(c):
        return obw / (c[0] / c[1])
    if prof.get("hires"):
        # hi-res sessions mimic the classic wide SDR++ view: a big FFT across a 1-2.4 MHz span, where narrow
        # signals are thin lines. (A 32k-65k FFT in a decimated 30-60 kHz span is a 0.5-2 s window -> smear.)
        wide = [c for c in combos if c[0] / c[1] >= 1e6 and fill(c) <= 0.85]
        if wide:
            return tuple(int(x) for x in wide[int(rng.integers(len(wide)))])
    good = [c for c in combos if 0.08 <= fill(c) <= 0.6]
    preferred = [c for c in good if c[0] in pref_sr and c[1] in pref_dec]
    pool = preferred or good
    if pool:
        return tuple(int(x) for x in pool[int(rng.integers(len(pool)))])
    best = min(combos, key=lambda c: abs(np.log(fill(c) / 0.3)))
    return int(best[0]), int(best[1])


def free_center(rng, obw: float) -> float:
    """433 MHz ISM is quiet here but only 1.74 MHz wide; 902-928 MHz is wide but busy with local traffic."""
    if obw > 1.6e6:
        return float(rng.uniform(905e6, 925e6))
    if obw > 0.9e6:
        return float(433.92e6 + rng.uniform(-20e3, 20e3))
    return float(rng.uniform(433.6e6, 434.2e6))


def run_class(label: str, sessions: int, variants: int, frames: int, seed: int, colormap: str | None = None,
              hires: bool | None = None):
    """``label`` is the dataset class; sessions draw a generator sub-type for it (e.g. morse -> OOK).
    colormap: SDR++ colormap name, "random" (any but Classic, drawn per session) or None (Classic).
    hires: force (True) or forbid (False) the big-FFT / fast-waterfall look; None = profile default."""
    forced = label if label in CLASS_OF else None  # e.g. "QPSK": capture that sub-type under its class
    label = CLASS_OF.get(label, label)
    rng = np.random.default_rng([seed, int(hashlib.md5(label.encode()).hexdigest()[:8], 16)])
    tx = Transmitter()
    IQ.mkdir(parents=True, exist_ok=True)
    for s in range(sessions):
        sub = forced or str(rng.choice(SUBTYPES.get(label, [label])))
        gen = GENERATORS[sub]
        cls = label
        prof = pick(sub, rng, hires=hires)
        if colormap == "random":  # walk a shuffled list so a class gets as many different maps as sessions
            others = [n for n in colormaps.luts() if n != "Classic"]
            cmap_order = cmap_order if s else [others[i] for i in rng.permutation(len(others))]
            CMAP["name"] = cmap_order[s % len(cmap_order)]
        else:
            CMAP["name"] = colormap or "Classic"
        # Pre-generate the first variant and zoom (sample rate / decimation) so it fills 8-60% of the screen
        t_est = screen_seconds(prof["fft_rate"], cal().get("height_px", 1191), prof["fft_size"])
        dur = float(min(max(t_est * 1.2, 2.0), 14.0))
        first_iq, first_meta = gen(rng, prof["gen_fs"], dur)
        obw0 = occupied_bw(first_iq, prof["gen_fs"])
        prof["rtl_sr"], prof["decimation"] = choose_zoom(sub, obw0, rng, prof)
        center = free_center(rng, obw0)
        prof_sdr = dict(freq=center, sample_rate=prof["rtl_sr"], gain=prof["rtl_gain"], fft_size=prof["fft_size"],
                        fft_rate=prof["fft_rate"], min_db=prof["min_db"], max_db=prof["max_db"],
                        decimation=prof["decimation"], range_db=prof["range_db"], colormap=CMAP["name"])
        prof_sdr = autolevel(prof_sdr)
        for _ in range(4):  # busy band (mostly 902-928 MHz): move until the TX-off screen is quiet
            time.sleep(min(screen_seconds(prof["fft_rate"], 1191), 6.0))
            base = signal_score(grab_crop("baseline"))
            if base < 25 or center < 900e6:  # 433 MHz: noise texture / a steady carrier is not "busy"
                break
            print(f"  {center / 1e6:.3f} MHz busy (baseline {base:.0f}), retuning", flush=True)
            center = free_center(rng, obw0)
            prof_sdr["freq"] = center
            sdr.restart_with(prof_sdr)
            time.sleep(2.5)
        prof.update(min_db=prof_sdr["min_db"], max_db=prof_sdr["max_db"])
        span = prof["rtl_sr"] / prof["decimation"]
        session_id = f"{cls}-{seed}-{s}"
        for v in range(variants):
            first = grab_crop("probe")
            t_screen = screen_seconds(prof["fft_rate"], first.height, prof["fft_size"])
            if v == 0:
                iq, meta = first_iq, first_meta
            else:
                for _ in range(6):  # later variants must also fit the chosen zoom
                    iq, meta = gen(rng, prof["gen_fs"], dur)
                    if occupied_bw(iq, prof["gen_fs"]) <= 0.85 * span:
                        break
            meta = {"subtype": sub, "hires": prof.get("hires", False), "overloaded": prof.get("overloaded", False), **meta}
            obw = min(occupied_bw(iq, prof["gen_fs"]), 0.95 * span)
            room = max(0.0, (0.9 * span - obw) / 2)          # keep the whole signal on screen
            offset = float(rng.uniform(-1, 1) * min(room, 0.35 * span))
            b_lo, b_hi = band_of(center)
            lo_ok, hi_ok = b_lo + obw / 2 + 20e3, b_hi - obw / 2 - 20e3
            if lo_ok > hi_ok:
                print(f"[{cls}] obw {obw/1e3:.0f} kHz does not fit the ISM band, variant skipped", flush=True)
                continue
            offset = float(np.clip(center + offset, lo_ok, hi_ok) - center)
            tx_fs, lo_rel, shift = tx_plan(span, offset)
            meta["obw_hz"] = int(obw)
            path = IQ / f"{session_id}-{v}.cs8"
            write_cs8(iq, prof["gen_fs"], tx_fs, shift, path)
            del iq
            gain = int(rng.integers(prof["tx_gain"][0], prof["tx_gain"][1] + 1))
            lo_hz = center + lo_rel - cal().get("freq_err_hz", 0) * center / 433.92e6  # error scales with freq (ppm)
            tx.start(path, lo_hz, tx_fs, gain)
            try:
                need = min(max(30.0, base + 20.0), 80.0)  # stand out from the TX-off baseline (capped: speckly noise)
                gain = settle_gain(tx, path, lo_hz, tx_fs, gain, t_screen, need=need)
                time.sleep(t_screen + 0.5)
                for k in range(frames):
                    if k:
                        time.sleep(t_screen)
                    img, score = grab_with_signal(f"{session_id}-{v}-{k}", t_screen, min_score=need)
                    if score < need - 10:
                        print(f"[{cls}] s{s} v{v} f{k} no visible signal (score {score:.0f}), skipped", flush=True)
                        continue
                    rel, h = save_png(img, cls)
                    log({"file": rel, "class": cls, "source": "synthetic", "group_id": f"{session_id}-{v}",
                         "session": session_id, "rtl_center_hz": int(center), "offset_hz": int(offset),
                         "span_hz": int(span), "decimation": prof["decimation"], "fft_size": prof["fft_size"],
                         "fft_rate": prof["fft_rate"], "rtl_gain": prof["rtl_gain"], "tx_gain": gain,
                         "min_db": prof["min_db"], "max_db": prof["max_db"], "tx_fs": int(tx_fs),
                         "colormap": CMAP["name"], "params": json.dumps(meta)})
                    print(f"[{cls}] s{s} v{v} f{k} score={score:.0f} t_screen={t_screen:.1f}s gain={gain} {meta}", flush=True)
            finally:
                tx.stop()
                path.unlink(missing_ok=True)
    tx.stop()
    CMAP["name"] = "Classic"


def calibrate(fft_rate=120, fft_size=16384, sample_rate=1_024_000, save=True):
    """Transmit a slow OOK pattern, take two screenshots dt apart, and find the vertical scroll."""
    sdr.restart_with(dict(freq=433.92e6, sample_rate=sample_rate, gain=20, fft_size=fft_size, fft_rate=fft_rate,
                          min_db=-80, max_db=-25))
    time.sleep(2)
    rng = np.random.default_rng(0)
    env = np.repeat((rng.random(400) > 0.5).astype(np.float32), int(0.05 * 250e3))
    tx_fs, lo_rel, shift = tx_plan(1.024e6, 100e3)
    path = IQ / "calib.cs8"
    IQ.mkdir(parents=True, exist_ok=True)
    write_cs8(env.astype(np.complex64), 250e3, tx_fs, shift, path)
    tx = Transmitter()
    tx.start(path, 433.92e6 + lo_rel, tx_fs, 20)
    try:
        time.sleep(4)
        a = np.asarray(grab_crop("c1").convert("L"), np.float32)
        t0 = time.time()
        time.sleep(1.0)
        b = np.asarray(grab_crop("c2").convert("L"), np.float32)
        dt = time.time() - t0
    finally:
        tx.stop()
        path.unlink(missing_ok=True)
    pa, pb = a.mean(1) - a.mean(), b.mean(1) - b.mean()
    best = max(range(1, len(pa) // 2), key=lambda s: np.corrcoef(pa[:-s], pb[s:])[0, 1])
    px_per_s = best / dt
    res = {"px_per_fft_line": px_per_s / fft_rate, "measured_fft_rate": fft_rate, "px_per_s": px_per_s,
           "height_px": len(pa), "fft_size": fft_size, "sample_rate": sample_rate}
    print(res)
    if save:
        c = cal()
        c.update({k: v for k, v in res.items() if k in ("px_per_fft_line", "measured_fft_rate", "px_per_s", "height_px")})
        CALIB.write_text(json.dumps(c, indent=2))
    return res


def _stop_all(*_):
    for t in Transmitter._live:
        t.stop()
    subprocess.run(["pkill", "-INT", "-x", "hackrf_transfer"], capture_output=True)


def _on_signal(signum, _frame):
    _stop_all()
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    import atexit
    import signal
    atexit.register(_stop_all)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    ap = argparse.ArgumentParser()
    ap.add_argument("classes", nargs="*")
    ap.add_argument("--sessions", type=int, default=25)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--colormap", help='SDR++ colormap name, or "random" (any but Classic, per session)')
    ap.add_argument("--hires", action="store_true", help="every session uses the big-FFT / fast-waterfall look")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    for c in a.classes:
        run_class(c, a.sessions, a.variants, a.frames, a.seed, colormap=a.colormap, hires=a.hires or None)


def measure_scroll(fft_rate: int, fft_size: int, sample_rate: int = 2_400_000) -> float:
    """True waterfall scroll speed (px/s) from ONE screenshot: transmit an on/off square wave of known
    period and measure its vertical period in pixels (no screenshot-timing dependence)."""
    period = max(0.004, 60.0 / (2.0 * fft_rate) / 1.0) * 2  # ~120 px per period at the nominal rate
    period = max(period, 4 * fft_size / sample_rate)       # keep the FFT window well inside a half period
    sdr.restart_with(dict(freq=433.92e6, sample_rate=sample_rate, gain=20, fft_size=fft_size, fft_rate=fft_rate,
                          min_db=-80, max_db=-25))
    time.sleep(2)
    fs = 250e3
    half = int(period / 2 * fs)
    env = np.tile(np.r_[np.ones(half), np.zeros(half)], int(6 / period) + 2).astype(np.complex64)
    tx_fs, lo_rel, shift = tx_plan(sample_rate, 200e3)
    path = IQ / "scroll.cs8"
    IQ.mkdir(parents=True, exist_ok=True)
    write_cs8(env, fs, tx_fs, shift, path)
    tx = Transmitter()
    tx.start(path, 433.92e6 + lo_rel - cal().get("freq_err_hz", 0), tx_fs, 15)
    try:
        time.sleep(4)
        a = np.asarray(grab_crop("scroll").convert("L"), np.float32)
    finally:
        tx.stop()
        path.unlink(missing_ok=True)
    col = a[:, int(a.shape[1] * (0.5 + 200e3 / sample_rate))]          # the carrier column
    col = np.convolve(col - col.mean(), np.ones(3) / 3, mode="same")
    ac = np.correlate(col, col, mode="full")[len(col) - 1:]
    ac /= ac[0] + 1e-9
    lag = int(np.argmax(ac[3:len(ac) // 2]) + 3)                       # first strong repeat = one period
    return lag / period
