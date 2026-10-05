"""NumPy baseband generators for every bench-synthesizable Classidyne waterfall class.

Every generator has the signature ``gen(rng, fs, dur) -> (iq, meta)``:
  * ``iq``   complex64 baseband centred on 0 Hz (the capture script adds the RF offset)
  * ``meta`` dict of the randomized parameters (written to manifest.csv)

Signals are built following the rules in aft-rfctf/tmp/generating-signals-with-numpy:
shaped amplitude edges, continuous phase for FSK (np.cumsum), RRC for linear modulations.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import oaconvolve

from iqlib import fftfilt, fm_modulate, lowpass_taps, rrc_taps, shift

AUDIO_FS = 48_000


# ----------------------------------------------------------------------------
# Shared building blocks
# ----------------------------------------------------------------------------

def _n(fs, dur):
    return int(round(fs * dur))


def resample(x: np.ndarray, fs_in: float, fs_out: float, n_out: int | None = None) -> np.ndarray:
    """Linear-interpolation resample (good enough for FM/AM message signals)."""
    n_out = n_out or int(len(x) * fs_out / fs_in)
    t_out = np.arange(n_out) / fs_out
    t_in = np.arange(len(x)) / fs_in
    return np.interp(t_out, t_in, x).astype(np.float32)


def shape_edges(env: np.ndarray, fs: float, ramp_s: float) -> np.ndarray:
    """Raised-cosine edges on an on/off envelope (prevents key clicks)."""
    r = max(3, int(ramp_s * fs)) | 1
    w = np.hanning(r)
    return oaconvolve(np.asarray(env, np.float32), (w / w.sum()).astype(np.float32), mode="same").astype(np.float32)


def cpfsk(freq_hz: np.ndarray, fs: float) -> np.ndarray:
    """Continuous-phase FSK from an instantaneous-frequency track."""
    return np.exp(1j * 2 * np.pi * np.cumsum(freq_hz) / fs).astype(np.complex64)


def gaussian_taps(bt: float, sps: int, span: int = 4) -> np.ndarray:
    t = (np.arange(span * sps + 1) - span * sps / 2) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    h = np.exp(-t ** 2 / (2 * sigma ** 2))
    return (h / h.sum()).astype(np.float32)


def symbols_to_track(sym: np.ndarray, sps: int, bt: float | None = None) -> np.ndarray:
    """Repeat symbols to samples, optionally Gaussian-filtered (GFSK/GMSK)."""
    x = np.repeat(sym.astype(np.float32), sps)
    if bt:
        x = oaconvolve(x, gaussian_taps(bt, sps), mode="same").astype(np.float32)
    return x


def gfsk(bits, rate, dev, fs, bt=0.5):
    sps = max(2, int(round(fs / rate)))
    track = symbols_to_track(2.0 * np.asarray(bits) - 1.0, sps, bt)
    return cpfsk(dev * track, fs)


def rrc_linear(symbols: np.ndarray, sym_rate: float, fs: float, beta: float) -> np.ndarray:
    """Pulse-shaped linear modulation (PSK/QAM/ASK) at sample rate fs."""
    sps = max(2, int(round(fs / sym_rate)))
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    return oaconvolve(up, rrc_taps(beta, sps, 10), mode="same").astype(np.complex64)


def ramp(burst: np.ndarray, fs: float, ramp_s: float = 20e-6) -> np.ndarray:
    """Raised-cosine power-up/down at the burst ends (real PAs ramp; avoids splatter)."""
    r = min(max(2, int(ramp_s * fs)), len(burst) // 2)
    w = np.hanning(2 * r).astype(np.float32)
    out = burst.astype(np.complex64).copy()
    out[:r] *= w[:r]
    out[-r:] *= w[r:]
    return out


def burst_train(burst: np.ndarray, fs: float, dur: float, gap_s, rng, jitter=0.3) -> np.ndarray:
    """Repeat a burst with (randomized) silent gaps until ``dur`` seconds are filled."""
    burst = ramp(burst, fs)
    out, total, n = [], 0, _n(fs, dur)
    lead = np.zeros(int(rng.uniform(0, gap_s) * fs), np.complex64)
    out.append(lead)
    total += len(lead)
    while total < n:
        out.append(burst)
        g = np.zeros(int(gap_s * rng.uniform(1 - jitter, 1 + jitter) * fs), np.complex64)
        out.append(g)
        total += len(burst) + len(g)
    return np.concatenate(out)[:n]


def fit(x: np.ndarray, fs: float, dur: float) -> np.ndarray:
    """Tile or trim to exactly dur seconds."""
    n = _n(fs, dur)
    if len(x) >= n:
        return x[:n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def rand_text(rng, n=12):
    alphabet = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
    return "".join(rng.choice(alphabet, n))


def speech_like(rng, dur, fs=AUDIO_FS) -> np.ndarray:
    """Synthetic voice: glottal pulse train with drifting pitch through formant resonances,
    gated into syllables and phrases. Bandwidth ~300-3400 Hz like real voice channels."""
    n = _n(fs, dur)
    t = np.arange(n) / fs
    f0 = rng.uniform(90, 220) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t))
    phase = 2 * np.pi * np.cumsum(f0) / fs
    src = np.zeros(n, np.float32)
    for k in range(1, 30):
        src += np.sin(k * phase) / k
    src += 0.3 * rng.standard_normal(n)
    out = np.zeros(n, np.float32)
    for fc, bw in [(rng.uniform(500, 900), 120), (rng.uniform(1100, 2000), 180), (rng.uniform(2300, 3200), 250)]:
        out += fftfilt(src, _bandpass(fc, bw * 3, fs)).astype(np.float32)
    syll = (rng.random(int(dur * 5) + 2) > 0.25).astype(np.float32)
    env = shape_edges(np.repeat(syll, fs // 5)[:n], fs, 0.03)
    phrase = (rng.random(int(dur / 1.5) + 2) > 0.3).astype(np.float32)
    env *= shape_edges(np.repeat(phrase, int(1.5 * fs))[:n], fs, 0.1)
    out *= env
    return (out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


def music_like(rng, dur, fs=AUDIO_FS) -> np.ndarray:
    """Broadband programme audio (chords + percussion noise) for broadcast FM/AM."""
    n = _n(fs, dur)
    t = np.arange(n) / fs
    x = np.zeros(n, np.float32)
    for _ in range(6):
        f = rng.uniform(80, 2000)
        x += np.sin(2 * np.pi * f * t) * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 2) * t))
    beat = np.repeat((rng.random(int(dur * 4) + 2) > 0.5).astype(np.float32), fs // 4)[:n]
    x += 2 * fftfilt(rng.standard_normal(n), lowpass_taps(12_000, fs, 101)) * shape_edges(beat, fs, 0.01)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def _bandpass(fc, bw, fs, ntaps=257):
    lp = lowpass_taps(bw / 2, fs, ntaps).astype(np.float64)
    n = np.arange(ntaps) - (ntaps - 1) / 2
    return (2 * lp * np.cos(2 * np.pi * fc * n / fs)).astype(np.float32)


def nbfm_from_audio(audio, dev, fs_audio, fs, dur):
    msg = resample(audio, fs_audio, fs, _n(fs, dur))
    return fm_modulate(msg, dev, fs)


def am_from_audio(audio, depth, fs_audio, fs, dur, carrier=True):
    msg = resample(audio, fs_audio, fs, _n(fs, dur))
    return ((1.0 if carrier else 0.0) + depth * msg).astype(np.complex64)


# ----------------------------------------------------------------------------
# Class generators
# ----------------------------------------------------------------------------

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....",
    "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.",
    "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}


def morse_units(text):
    units = []
    for w, word in enumerate(text.split(" ")):
        if w:
            units += [0] * 7
        for i, ch in enumerate(word):
            if i:
                units += [0] * 3
            for j, s in enumerate(MORSE[ch]):
                if j:
                    units += [0]
                units += [1] if s == "." else [1, 1, 1]
    return np.array(units + [0] * 7, np.float32)


def gen_morse(rng, fs, dur):
    wpm = rng.uniform(8, 30)
    text = " ".join(rand_text(rng, rng.integers(3, 7)) for _ in range(4))
    env = np.repeat(morse_units(text), int(1.2 / wpm * fs))
    env = shape_edges(env, fs, rng.uniform(0.002, 0.008))
    return fit(env.astype(np.complex64), fs, dur), {"wpm": round(wpm, 1), "text": text}


def _ook_packet(rng, fs, rate, manchester):
    bits = np.concatenate([np.tile([1, 0], 16), [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                           rng.integers(0, 2, int(rng.integers(24, 96)))])
    if manchester:
        bits = np.ravel(np.column_stack([bits, 1 - bits]))
        rate *= 2
    env = np.repeat(bits.astype(np.float32), max(1, int(fs / rate)))
    return shape_edges(env, fs, min(50e-6, 0.2 / rate)).astype(np.complex64)


def gen_OOK(rng, fs, dur):
    rate = float(rng.choice([300, 500, 1000, 2000, 4000, 8000]))
    manch = bool(rng.random() < 0.5)
    pkt = _ook_packet(rng, fs, rate, manch)
    reps = int(rng.integers(2, 6))
    block = np.concatenate([np.concatenate([pkt, np.zeros(int(len(pkt) * 0.3), np.complex64)])] * reps)
    gap = rng.uniform(0.05, 0.6)
    return burst_train(block, fs, dur, gap, rng), {"bit_rate": rate, "manchester": manch, "repeats": reps}


def gen_2ASK(rng, fs, dur):
    rate = float(rng.choice([1e3, 2.4e3, 4.8e3, 9.6e3, 19.2e3, 50e3]))
    low = rng.uniform(0.0, 0.4)
    nsym = int(dur * rate) + 20
    sym = np.where(rng.integers(0, 2, nsym) == 1, 1.0, low).astype(np.complex64)
    iq = rrc_linear(sym, rate, fs, rng.uniform(0.3, 0.8)) if rate > 5e3 else \
        shape_edges(np.repeat(sym.real, int(fs / rate)), fs, 0.15 / rate).astype(np.complex64)
    bursty = rng.random() < 0.5
    iq = fit(iq, fs, dur)
    if bursty:
        on = np.repeat((rng.random(int(dur * 4) + 2) > 0.35).astype(np.float32), int(fs / 4))[:len(iq)]
        iq = iq * shape_edges(on, fs, 0.002)
    return iq.astype(np.complex64), {"sym_rate": rate, "low_level": round(low, 2), "bursty": bursty}


def gen_2FSK(rng, fs, dur):
    """Binary FSK: continuous telemetry, ISM sensor packets (preamble + sync + payload) or FSK keyfob words.
    Low rates are favoured so the tone alternation stays visible on fast, high-resolution waterfalls."""
    rate = float(rng.choice([100, 200, 300, 600, 1200, 2400, 4800, 9600, 19200, 38400],
                            p=[.12, .12, .12, .12, .12, .1, .1, .08, .06, .06]))
    dev = float(np.clip(rate * rng.uniform(0.5, 3.0), rng.uniform(3e3, 20e3), 60e3))
    bt = rng.choice([None, None, 0.5, 1.0])
    kind = str(rng.choice(["continuous", "packets", "keyfob"], p=[0.35, 0.4, 0.25]))
    if kind == "continuous":
        iq = fit(gfsk(rng.integers(0, 2, int(dur * rate) + 8), rate, dev, fs, bt=bt), fs, dur)
    else:
        def word():
            nbits = int(rng.integers(48, 160)) if kind == "packets" else int(rng.integers(40, 90))
            b = np.concatenate([np.tile([1, 0], int(rng.integers(8, 32))), [1, 1, 0, 1, 0, 0, 1, 1], rng.integers(0, 2, nbits)])
            return ramp(gfsk(b, rate, dev, fs, bt=bt), fs, min(2e-3, 0.5 / rate))
        w = word()
        if kind == "keyfob":
            w = np.concatenate([np.concatenate([w, np.zeros(int(rng.uniform(5e-3, 30e-3) * fs), np.complex64)])
                                for _ in range(int(rng.integers(3, 8)))])
        iq = burst_train(w, fs, dur, rng.uniform(0.05, 1.0) * max(1.0, len(w) / fs), rng)
    return iq.astype(np.complex64), {"kind": kind, "bit_rate": rate, "dev_hz": round(dev), "bt": bt}


def gen_4FSK(rng, fs, dur):
    rate = float(rng.choice([1.2e3, 2.4e3, 4.8e3, 9.6e3, 19.2e3]))
    dev = rate * rng.uniform(0.5, 1.5)
    sym = rng.choice([-3, -1, 1, 3], int(dur * rate) + 10).astype(np.float32)
    sps = max(2, int(fs / rate))
    track = symbols_to_track(sym, sps, bt=rng.choice([None, 0.5]))
    return fit(cpfsk(dev / 3 * track, fs), fs, dur), {"sym_rate": rate, "outer_dev_hz": round(dev)}


def _linear(rng, fs, dur, constellation, name):
    rate = float(rng.choice([25e3, 50e3, 100e3, 125e3, 250e3, 500e3]))
    beta = rng.uniform(0.2, 0.5)
    sym = constellation[rng.integers(0, len(constellation), int(dur * rate) + 50)]
    iq = fit(rrc_linear(sym.astype(np.complex64), rate, fs, beta), fs, dur)
    bursty = rng.random() < 0.4
    if bursty:
        on = np.repeat((rng.random(int(dur * 8) + 2) > 0.4).astype(np.float32), int(fs / 8))[:len(iq)]
        iq = iq * shape_edges(on, fs, 0.0005)
    return iq.astype(np.complex64), {"modulation": name, "sym_rate": rate, "beta": round(beta, 2), "bursty": bursty}


def _qam(m):
    side = int(np.sqrt(m))
    if side * side == m:
        pts = [complex(i, q) for i in range(-side + 1, side, 2) for q in range(-side + 1, side, 2)]
    else:  # 32-QAM cross: 6x6 grid minus the 4 corners
        pts = [complex(i, q) for i in range(-5, 6, 2) for q in range(-5, 6, 2) if not (abs(i) == 5 and abs(q) == 5)]
    pts = np.array(pts)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def gen_BPSK(rng, fs, dur):
    return _linear(rng, fs, dur, np.array([1, -1], np.complex64), "BPSK")


def gen_QPSK(rng, fs, dur):
    return _linear(rng, fs, dur, np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4))), "QPSK")


def gen_8PSK(rng, fs, dur):
    return _linear(rng, fs, dur, np.exp(2j * np.pi * np.arange(8) / 8), "8PSK")


def gen_16QAM(rng, fs, dur):
    return _linear(rng, fs, dur, _qam(16), "16QAM")


def gen_32QAM(rng, fs, dur):
    return _linear(rng, fs, dur, _qam(32), "32QAM")


def gen_am(rng, fs, dur):
    audio = music_like(rng, dur) if rng.random() < 0.6 else speech_like(rng, dur)
    audio = fftfilt(audio, lowpass_taps(rng.uniform(4500, 9000), AUDIO_FS, 201)).astype(np.float32)
    depth = rng.uniform(0.4, 0.95)
    return am_from_audio(audio, depth, AUDIO_FS, fs, dur), {"depth": round(depth, 2), "type": "broadcast"}


def gen_airband(rng, fs, dur):
    """Push-to-talk AM voice: carrier keys up for a transmission, then drops."""
    n = _n(fs, dur)
    audio = speech_like(rng, dur)
    key = np.zeros(int(dur * 10) + 2, np.float32)
    i = 0
    while i < len(key):
        L = int(rng.uniform(15, 50))
        key[i:i + L] = 1
        i += L + int(rng.uniform(5, 30))
    key_env = shape_edges(np.repeat(key, AUDIO_FS // 10)[: len(audio)], AUDIO_FS, 0.02)
    iq = am_from_audio(audio * key_env, rng.uniform(0.6, 0.9), AUDIO_FS, fs, dur, carrier=False)
    iq = iq + resample(key_env, AUDIO_FS, fs, n)
    return iq.astype(np.complex64), {"type": "ptt-am"}


def gen_fm(rng, fs, dur):
    """WBFM broadcast multiplex: mono + 19 kHz pilot + 38 kHz L-R + 57 kHz RDS."""
    fa = 200_000
    n = int(dur * fa)
    t = np.arange(n) / fa
    left, right = (resample(music_like(rng, dur), AUDIO_FS, fa, n) for _ in range(2))
    rds = np.repeat(rng.choice([-1.0, 1.0], int(dur * 1187.5) + 2), int(fa / 1187.5) + 1)[:n]
    mpx = 0.45 * (left + right) / 2 + 0.08 * np.sin(2 * np.pi * 19e3 * t) \
        + 0.35 * (left - right) / 2 * np.sin(2 * np.pi * 38e3 * t) + 0.04 * rds * np.sin(2 * np.pi * 57e3 * t)
    stereo = rng.random() < 0.8
    if not stereo:
        mpx = 0.9 * (left + right) / 2
    msg = resample(mpx / np.max(np.abs(mpx)), fa, fs, _n(fs, dur))
    return fm_modulate(msg, 75_000, fs), {"stereo": stereo, "dev_hz": 75000}


def _pocsag_bits(rng):
    pre = np.tile([1, 0], 288)
    sync = np.unpackbits(np.array([0x7C, 0xD2, 0x15, 0xD8], np.uint8))
    batches = [pre]
    for _ in range(int(rng.integers(1, 4))):
        batches += [sync, rng.integers(0, 2, 16 * 32)]
    return np.concatenate(batches)


def gen_pocsag(rng, fs, dur):
    rate = float(rng.choice([512, 1200, 2400]))
    bits = _pocsag_bits(rng)
    burst = gfsk(bits, rate, 4500, fs, bt=None)
    return burst_train(burst, fs, dur, rng.uniform(0.3, 2.0), rng), {"baud": rate, "dev_hz": 4500}


def _adsb_msg(fs):
    chips_per_us = 2
    pre = np.zeros(16)
    pre[[0, 2, 7, 9]] = 1
    return pre, chips_per_us


def gen_ads_b(rng, fs, dur):
    """Mode S 1090ES: 8 us preamble + 112 PPM bits (1 Mb/s), many aircraft => random bursts."""
    spc = max(1, int(fs / 2e6))  # samples per 0.5 us chip
    pre = np.zeros(16, np.float32)
    pre[[0, 2, 7, 9]] = 1
    n = _n(fs, dur)
    env = np.zeros(n, np.float32)
    rate = rng.uniform(80, 600)  # messages/s (busy vs. quiet sky)
    t = rng.exponential(1 / rate)
    amps = rng.uniform(0.5, 1.0, 50)  # each aircraft has its own signal level
    while t < dur - 200e-6:
        bits = rng.integers(0, 2, int(rng.choice([56, 112])))
        chips = np.ravel(np.column_stack([bits, 1 - bits])).astype(np.float32)
        frame = np.repeat(np.concatenate([pre, chips]), spc) * rng.choice(amps)
        i = int(t * fs)
        env[i:i + len(frame)] = np.maximum(env[i:i + len(frame)], frame[: n - i])
        t += rng.exponential(1 / rate)
    iq = shape_edges(env, fs, 50e-9).astype(np.complex64)
    # band-limit to ~1.5 MHz (what a 2.4 MS/s RTL shows of 1090ES anyway) so it fits the 433 ISM band
    iq = oaconvolve(iq, lowpass_taps(min(750e3, 0.45 * fs), fs, 129), mode="same").astype(np.complex64)
    return iq, {"msg_rate": round(rate)}


def gen_ais(rng, fs, dur):
    """GMSK 9600 b/s, BT 0.4, 26.7 ms SOTDMA bursts alternating between two channels 50 kHz apart."""
    n = _n(fs, dur)
    iq = np.zeros(n, np.complex64)
    t = rng.uniform(0, 0.1)
    rate = rng.uniform(4, 40)
    while t < dur - 0.03:
        bits = np.concatenate([np.tile([0, 1], 12), [0, 1, 1, 1, 1, 1, 1, 0], rng.integers(0, 2, 184)])
        burst = ramp(gfsk(bits, 9600, 2400, fs, bt=0.4), fs, 200e-6) * rng.uniform(0.2, 1.0)
        burst = shift(burst, rng.choice([-25_000, 25_000]), fs)
        i = int(t * fs)
        m = min(len(burst), n - i)
        iq[i:i + m] += burst[:m]
        t += rng.exponential(1 / rate)
    return iq, {"bursts_per_s": round(rate, 1)}


def _afsk1200(rng, nbits):
    bits = np.concatenate([np.tile([0, 1, 1, 1, 1, 1, 1, 0], int(rng.integers(10, 40))), rng.integers(0, 2, nbits)])
    tones = np.where(np.repeat(bits, AUDIO_FS // 1200) == 1, 1200.0, 2200.0)
    return np.sin(2 * np.pi * np.cumsum(tones) / AUDIO_FS).astype(np.float32)


def gen_packet(rng, fs, dur):
    """AX.25/APRS: AFSK1200 audio on NBFM, sporadic packet bursts."""
    n = int(dur * AUDIO_FS)
    audio = np.zeros(n, np.float32)
    key = np.zeros(n, np.float32)
    t = rng.uniform(0, 0.5)
    while t < dur - 0.3:
        a = _afsk1200(rng, int(rng.integers(200, 1200)))
        i = int(t * AUDIO_FS)
        m = min(len(a), n - i)
        audio[i:i + m] = a[:m]
        key[i:i + m] = 1
        t += m / AUDIO_FS + rng.uniform(0.2, 2.5)
    dev = rng.uniform(2500, 4000)
    carrier = nbfm_from_audio(audio, dev, AUDIO_FS, fs, dur)
    return (carrier * resample(shape_edges(key, AUDIO_FS, 0.002), AUDIO_FS, fs, len(carrier))).astype(np.complex64), \
        {"dev_hz": round(dev)}


def gen_Radioteletype(rng, fs, dur):
    baud, shift_hz = [(45.45, 170), (50, 450), (75, 850), (50, 85)][rng.integers(0, 4)]
    bits = rng.integers(0, 2, int(dur * baud) + 5)
    sps = int(fs / baud)
    return fit(cpfsk(shift_hz / 2 * (2.0 * np.repeat(bits, sps) - 1), fs), fs, dur), \
        {"baud": baud, "shift_hz": shift_hz}


def gen_sstv(rng, fs, dur):
    """Martin/Scottie-like scan lines: 1200 Hz sync then 1500-2300 Hz pixel tones on NBFM."""
    line_ms = rng.choice([146.4, 226.8, 138.2, 428.2])
    n = int(dur * AUDIO_FS)
    freq = np.empty(0, np.float32)
    img_seed = rng.random(320)
    k = 0
    while len(freq) < n:
        sync = np.full(int(0.0048 * AUDIO_FS), 1200.0)
        k += 1
        row = 1500 + 800 * np.clip(img_seed + 0.3 * np.sin(k / 20 + np.arange(320) / 30), 0, 1)
        pix = np.repeat(row, max(1, int(line_ms / 1000 * AUDIO_FS / 320)))
        freq = np.concatenate([freq, sync, pix])
    audio = np.sin(2 * np.pi * np.cumsum(freq[:n]) / AUDIO_FS).astype(np.float32)
    return nbfm_from_audio(audio, rng.uniform(2500, 5000), AUDIO_FS, fs, dur), {"line_ms": float(line_ms)}


def gen_automatic_picture_transmission(rng, fs, dur):
    """NOAA APT: 2 lines/s, 2400 Hz AM subcarrier carrying sync A/B + image, FM +/-17 kHz."""
    words_per_line = 4160 // 2
    n = int(dur * AUDIO_FS)
    lines = int(dur * 2) + 2
    img = np.clip(0.5 + 0.4 * np.sin(np.linspace(0, 6, words_per_line))[None, :] * rng.uniform(0.5, 1, (lines, 1))
                  + 0.1 * rng.standard_normal((lines, words_per_line)), 0, 1)
    sync_a = np.tile([1, 1, 0, 0], 7 * 4)[:39 * 2]
    img[:, :len(sync_a)] = sync_a
    img[:, words_per_line // 2: words_per_line // 2 + 39] = np.tile([1, 1, 1, 0, 0], 8)[:39]
    env = resample(img.ravel(), words_per_line * 2, AUDIO_FS, n)
    t = np.arange(n) / AUDIO_FS
    audio = (0.1 + 0.9 * env) * np.sin(2 * np.pi * 2400 * t)
    return nbfm_from_audio(audio.astype(np.float32), 17_000, AUDIO_FS, fs, dur), {"dev_hz": 17000}


def _chirp(sf, bw, fs, sym, up=True):
    N = 2 ** sf
    sps = int(fs / bw * N)
    t = np.arange(sps) / fs
    T = N / bw
    f0 = (sym / N) * bw
    f = ((f0 + bw * t / T) % bw) - bw / 2
    if not up:
        f = -f
    return np.exp(1j * 2 * np.pi * np.cumsum(f) / fs).astype(np.complex64)


def gen_lora(rng, fs, dur):
    bw = float(rng.choice([125e3, 250e3, 500e3]))
    # keep a symbol >= ~3 ms so the chirp slope is resolvable on a waterfall
    sf = int(rng.integers(max(7, int(np.ceil(np.log2(0.003 * bw)))), 13))
    N = 2 ** sf
    pre = [_chirp(sf, bw, fs, 0)] * int(rng.integers(6, 10))
    sfd = [_chirp(sf, bw, fs, 0, up=False)] * 2 + [_chirp(sf, bw, fs, 0, up=False)[: int(fs / bw * N / 4)]]
    payload = [_chirp(sf, bw, fs, int(s)) for s in rng.integers(0, N, int(rng.integers(8, 40)))]
    pkt = np.concatenate(pre + sfd + payload)
    return burst_train(pkt, fs, dur, rng.uniform(0.1, 1.5), rng), {"sf": sf, "bw_hz": bw}


def gen_RS41_Radiosonde(rng, fs, dur):
    """Vaisala RS41: GFSK 4800 Bd, ~2.4 kHz deviation, 1 frame/s, short gap between frames."""
    frame_s = rng.uniform(0.6, 0.9)
    frame = gfsk(rng.integers(0, 2, int(4800 * frame_s)), 4800, 2400, fs, bt=0.5)
    carrier_gap = rng.random() < 0.5  # some sondes keep the carrier between frames
    if carrier_gap:
        frame = np.concatenate([frame, np.exp(1j * np.zeros(int((1 - frame_s) * fs))).astype(np.complex64)])
        return fit(frame, fs, dur), {"frame_s": round(frame_s, 2), "carrier_gap": True}
    return burst_train(frame, fs, dur, 1 - frame_s, rng, jitter=0.02), {"frame_s": round(frame_s, 2), "carrier_gap": False}


def gen_remote_keyless_entry(rng, fs, dur):
    """Keyfob: PWM OOK (KeeLoq-like preamble + 66-bit code word) or 2FSK, repeated per button press."""
    te = rng.uniform(200e-6, 500e-6)
    if True:  # OOK only: keyfobs are filed under OOK (FSK fobs are a 2FSK sub-type)
        pre = np.tile([1, 0], 12)
        bits = rng.integers(0, 2, 66)
        pwm = np.concatenate([[1, 0, 0] if b else [1, 1, 0] for b in bits])
        sym = np.concatenate([pre, np.zeros(10), pwm]).astype(np.float32)
        word = shape_edges(np.repeat(sym, int(te * fs)), fs, 10e-6).astype(np.complex64)
        kind = "ook-pwm"
    else:
        bits = np.concatenate([np.tile([1, 0], 16), rng.integers(0, 2, 80)])
        word = gfsk(bits, 1 / te, rng.uniform(15e3, 40e3), fs, bt=None)
        kind = "2fsk"
    reps = int(rng.integers(3, 8))
    press = np.concatenate([np.concatenate([word, np.zeros(int(0.015 * fs), np.complex64)])] * reps)
    return burst_train(press, fs, dur, rng.uniform(0.4, 2.0), rng), {"te_us": round(te * 1e6), "kind": kind, "repeats": reps}


def gen_z_wave(rng, fs, dur):
    """Z-Wave R1 (9.6k Manchester FSK), R2 (40k NRZ FSK), R3 (100k GFSK); short frames + ACKs."""
    rate, dev, bt, manch = [(9600, 20e3, None, True), (40e3, 20e3, None, False), (100e3, 29e3, 0.6, False)][rng.integers(0, 3)]
    def frame(nbytes):
        b = np.concatenate([np.tile([0, 1], 40), [1, 1, 1, 1, 0, 0, 0, 0], rng.integers(0, 2, nbytes * 8)])
        if manch:
            b = np.ravel(np.column_stack([b, 1 - b]))
        return gfsk(b, rate * (2 if manch else 1), dev, fs, bt=bt)
    txn = np.concatenate([ramp(frame(int(rng.integers(10, 40))), fs), np.zeros(int(0.003 * fs), np.complex64),
                          ramp(frame(10), fs) * rng.uniform(0.3, 1.0)])
    return burst_train(txn, fs, dur, rng.uniform(0.1, 1.0), rng), {"rate": rate, "dev_hz": dev}


def gen_digital_speech_decoder(rng, fs, dur):
    """DMR (4FSK 4800 sym/s, 30 ms TDMA bursts) or P25 Phase 1 C4FM (continuous)."""
    if rng.random() < 0.6:
        sym = rng.choice([-3, -1, 1, 3], int(dur * 4800) + 10)
        track = symbols_to_track(sym, int(fs / 4800), bt=0.5)
        iq = fit(cpfsk(648 * track, fs), fs, dur)
        slot = np.tile(np.concatenate([np.ones(int(0.0275 * fs)), np.zeros(int(0.0325 * fs))]), int(dur / 0.06) + 2)[:len(iq)]
        both = rng.random() < 0.3  # repeater with both timeslots active looks continuous
        iq = iq if both else iq * shape_edges(slot.astype(np.float32), fs, 0.0005)
        meta = {"mode": "DMR", "both_slots": both}
    else:
        sym = rng.choice([-3, -1, 1, 3], int(dur * 4800) + 10)
        track = symbols_to_track(sym, int(fs / 4800), bt=0.5)
        iq = fit(cpfsk(600 * track, fs), fs, dur)
        meta = {"mode": "P25-C4FM"}
    # PTT: the user talks in overs
    key = (rng.random(int(dur / 2) + 2) > 0.3).astype(np.float32)
    key[0] = 1
    key = np.repeat(key, int(2 * fs))[:len(iq)]
    return (iq * shape_edges(key, fs, 0.005)).astype(np.complex64), meta


def gen_vor(rng, fs, dur):
    """VOR: AM carrier + 30 Hz variable AM + 9960 Hz subcarrier FM'd +/-480 Hz at 30 Hz + 1020 Hz Morse ID."""
    n = int(dur * AUDIO_FS)
    t = np.arange(n) / AUDIO_FS
    ph = rng.uniform(0, 2 * np.pi)
    var30 = 0.3 * np.cos(2 * np.pi * 30 * t + ph)
    sub = 0.3 * np.cos(2 * np.pi * 9960 * t + 480 / 30 * np.sin(2 * np.pi * 30 * t))
    ident = np.repeat(morse_units(rand_text(rng, 3)), int(0.1 * AUDIO_FS))
    ident = np.pad(ident, (0, max(0, n - len(ident))))[:n]
    idt = 0.1 * shape_edges(ident, AUDIO_FS, 0.005) * np.sin(2 * np.pi * 1020 * t)
    audio = var30 + sub + idt
    return am_from_audio(audio.astype(np.float32), 1.0, AUDIO_FS, fs, dur), {"ident": True}


def _ofdm(rng, fs, dur, n_sc, spacing, cp_frac, frame_s=None, null_s=0.0, alloc=None):
    """Generic OFDM symbol stream (DQPSK/QPSK subcarriers) resampled to fs."""
    nfft = 1 << int(np.ceil(np.log2(n_sc * 1.4)))
    fs_o = nfft * spacing
    cp = int(nfft * cp_frac)
    syms = []
    total = 0
    n_target = int(dur * fs_o) + nfft
    k = 0
    while total < n_target:
        X = np.zeros(nfft, np.complex64)
        active = np.r_[-(n_sc // 2):0, 1:n_sc // 2 + 1]
        mask = np.ones(len(active), bool) if alloc is None else alloc(k, len(active))
        X[active[mask] % nfft] = np.exp(1j * np.pi / 2 * rng.integers(0, 4, mask.sum()) + 1j * np.pi / 4)
        x = np.fft.ifft(X)
        s = np.concatenate([x[-cp:], x])
        syms.append(s)
        total += len(s)
        k += 1
        if frame_s and null_s and total % int(frame_s * fs_o) < len(s):
            z = np.zeros(int(null_s * fs_o), np.complex64)
            syms.append(z)
            total += len(z)
    y = np.concatenate(syms).astype(np.complex64)
    from fractions import Fraction
    from scipy.signal import resample_poly
    fr = Fraction(int(fs), int(fs_o)).limit_denominator(1000)
    out = resample_poly(y, fr.numerator, fr.denominator)
    return fit(out.astype(np.complex64), fs, dur)


def gen_digital_audio_broadcasting(rng, fs, dur):
    """DAB Mode I: 1536 subcarriers @ 1 kHz (1.536 MHz), Tg = 246 us, 1 ms null symbol every 96 ms."""
    iq = _ofdm(rng, fs, dur, 1536, 1000, 0.246, frame_s=0.096, null_s=0.001)
    return iq, {"mode": "I", "bw_hz": 1_536_000}


def gen_cellular(rng, fs, dur):
    if rng.random() < 0.5:
        # LTE 1.4/3 MHz downlink: subframes with varying resource-block allocations
        n_rb = 6  # LTE 1.4 MHz (1.08 MHz occupied) is the only LTE bandwidth that fits the RTL span
        n_sc = 12 * n_rb
        load = rng.uniform(0.2, 1.0)
        cache = {}
        def alloc(k, n):
            sf = k // 14  # resource blocks are scheduled per 1 ms subframe
            if k % 14 < int(rng.integers(1, 4)) or k % 70 < 2:
                return np.ones(n, bool)                     # PDCCH control region / sync symbols span the band
            if sf not in cache:
                cache[sf] = np.repeat(rng.random(n_rb) < load, 12)
            blk = cache[sf]
            return blk[:n] if len(blk) >= n else np.pad(blk, (0, n - len(blk)))
        iq = _ofdm(rng, fs, dur, n_sc, 15_000, 0.07, alloc=alloc)
        return iq, {"tech": "LTE", "n_rb": n_rb, "load": round(load, 2)}
    # GSM: several 200 kHz GMSK carriers; BCCH continuous, traffic carriers in 577 us timeslots
    n = _n(fs, dur)
    iq = np.zeros(n, np.complex64)
    n_carr = int(rng.integers(1, 5))
    offs = rng.choice(np.arange(-3, 4) * 200e3, n_carr, replace=False)
    for c, off in enumerate(offs):
        bits = rng.integers(0, 2, int(dur * 270_833) + 10)
        sig = fit(gfsk(bits, 270_833, 270_833 / 4, fs, bt=0.3), fs, dur)
        if c:  # traffic carrier: only some timeslots active
            slots = rng.random(8) < rng.uniform(0.2, 0.8)
            frame = np.repeat(slots.astype(np.float32), int(577e-6 * fs))
            sig = sig * shape_edges(fit(frame.astype(np.complex64), fs, dur).real, fs, 10e-6)
        iq += shift(sig, off, fs) * rng.uniform(0.3, 1.0)
    return iq, {"tech": "GSM", "carriers": n_carr}


def gen_bluetooth(rng, fs, dur):
    """Classic BR/BLE-like GFSK 1 Mb/s bursts hopping across 1 MHz channels in the visible span."""
    n = _n(fs, dur)
    iq = np.zeros(n, np.complex64)
    span = min(fs * 0.8, 0.5e6)  # hop centres; with 1 MHz bursts the total stays inside the 433 ISM band
    chans = np.linspace(-span / 2, span / 2, 3)
    t = 0.0
    slot = 625e-6
    while t < dur - 0.003:
        L = int(rng.choice([1, 1, 1, 3, 5]))
        nb = int((L * slot - 220e-6) * 1e6)
        burst = ramp(gfsk(rng.integers(0, 2, max(nb, 50)), 1e6, rng.uniform(140e3, 175e3), fs, bt=0.5), fs, 4e-6)
        burst = shift(burst, rng.choice(chans), fs) * rng.uniform(0.3, 1.0)
        i = int(t * fs)
        m = min(len(burst), n - i)
        if rng.random() < 0.6:
            iq[i:i + m] += burst[:m]
        t += L * slot
    return iq, {"type": "br-hopping"}


# Sub-types that are filed under another class label (each keeps its own display profile).
# PSK and QAM share one waterfall class: RRC-shaped linear modulations differ only in constellation,
# which a spectrogram cannot show (same bandwidth = Rs * (1 + beta)).
CLASS_OF = {"morse": "OOK", "remote-keyless-entry": "OOK",
            "BPSK": "psk-qam", "QPSK": "psk-qam", "8PSK": "psk-qam", "16QAM": "psk-qam", "32QAM": "psk-qam"}
SUBTYPES = {"OOK": ["OOK", "morse", "remote-keyless-entry"],
            "psk-qam": ["BPSK", "QPSK", "8PSK", "16QAM", "32QAM"]}

GENERATORS = {
    "morse": gen_morse,
    "OOK": gen_OOK,
    "2ASK": gen_2ASK,
    "2FSK": gen_2FSK,
    "4FSK": gen_4FSK,
    "BPSK": gen_BPSK,
    "QPSK": gen_QPSK,
    "8PSK": gen_8PSK,
    "16QAM": gen_16QAM,
    "32QAM": gen_32QAM,
    "am": gen_am,
    "airband": gen_airband,
    "fm": gen_fm,
    "pocsag": gen_pocsag,
    "ads-b": gen_ads_b,
    "ais": gen_ais,
    "packet": gen_packet,
    "Radioteletype": gen_Radioteletype,
    "sstv": gen_sstv,
    "automatic-picture-transmission": gen_automatic_picture_transmission,
    "lora": gen_lora,
    "RS41-Radiosonde": gen_RS41_Radiosonde,
    "remote-keyless-entry": gen_remote_keyless_entry,
    "z-wave": gen_z_wave,
    "digital-speech-decoder": gen_digital_speech_decoder,
    "vor": gen_vor,
    "digital-audio-broadcasting": gen_digital_audio_broadcasting,
    "cellular": gen_cellular,
    "bluetooth": gen_bluetooth,
}
