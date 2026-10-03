"""Shared helpers for the iq_recordings/ challenges (instructor-side only).

Every challenge's generate.py and reference solver import this module with:

    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))
    import iqlib

It covers the four container formats the challenges ship in, a few DSP
building blocks (FIR design, frequency shifting, AWGN) and the CRCs used in
the framing. NumPy only: the venv has no SciPy.
"""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Container formats
# --------------------------------------------------------------------------

def _normalise(iq: np.ndarray, peak: float) -> np.ndarray:
    m = np.max(np.abs(np.concatenate([iq.real, iq.imag])))
    return iq / m * peak if m > 0 else iq


def write_wav_iq(path, iq: np.ndarray, fs: int, peak: float = 0.7) -> None:
    """Stereo 16-bit WAV, I = left, Q = right (SDR++, SDRangel, GQRX style)."""
    iq = _normalise(iq, peak)
    out = np.empty(2 * len(iq), dtype=np.int16)
    out[0::2] = np.clip(np.round(iq.real * 32767), -32767, 32767)
    out[1::2] = np.clip(np.round(iq.imag * 32767), -32767, 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(out.tobytes())


def read_wav_iq(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        assert w.getnchannels() == 2 and w.getsampwidth() == 2
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    iq = (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 32768
    return iq.astype(np.complex64), fs


def write_cf32(path, iq: np.ndarray, peak: float = 0.7) -> None:
    """Raw interleaved float32 I/Q (GNU Radio 'complex' file, inspectrum .cf32)."""
    _normalise(iq, peak).astype(np.complex64).tofile(str(path))


def read_cf32(path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.complex64)


def write_cu8(path, iq: np.ndarray, peak: float = 0.8) -> None:
    """Raw interleaved unsigned 8-bit I/Q, offset 127.5 (rtl_sdr native)."""
    iq = _normalise(iq, peak) * 127.5
    out = np.empty(2 * len(iq), dtype=np.uint8)
    out[0::2] = np.clip(np.round(iq.real + 127.5), 0, 255)
    out[1::2] = np.clip(np.round(iq.imag + 127.5), 0, 255)
    out.tofile(str(path))


def read_cu8(path) -> np.ndarray:
    d = np.fromfile(str(path), dtype=np.uint8).astype(np.float32)
    return ((d[0::2] - 127.5) + 1j * (d[1::2] - 127.5)).astype(np.complex64) / 127.5


def write_sigmf(base, iq: np.ndarray, fs: float, peak: float = 0.7) -> tuple[Path, Path]:
    """SigMF recording: <base>.sigmf-data (cf32_le) + <base>.sigmf-meta (JSON).

    Only the datatype and sample rate are written. No center frequency and
    no description: students get nothing that hints at the band or protocol.
    """
    base = Path(base)
    data_path = base.with_suffix(".sigmf-data")
    meta_path = base.with_suffix(".sigmf-meta")
    write_cf32(data_path, iq, peak)
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(fs),
            "core:version": "1.0.0",
            "core:sha512": hashlib.sha512(data_path.read_bytes()).hexdigest(),
        },
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return data_path, meta_path


def read_sigmf(base) -> tuple[np.ndarray, float, None]:
    base = Path(base)
    meta = json.loads(base.with_suffix(".sigmf-meta").read_text())
    assert meta["global"]["core:datatype"] == "cf32_le"
    iq = read_cf32(base.with_suffix(".sigmf-data"))
    return iq, meta["global"]["core:sample_rate"], None


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# DSP building blocks
# --------------------------------------------------------------------------

def shift(iq: np.ndarray, f_hz: float, fs: float, phase: float = 0.0) -> np.ndarray:
    n = np.arange(len(iq))
    return (iq * np.exp(1j * (2 * np.pi * f_hz * n / fs + phase))).astype(np.complex64)


def lowpass_taps(cutoff_hz: float, fs: float, ntaps: int = 129) -> np.ndarray:
    """Windowed-sinc low-pass FIR (Blackman), unity DC gain."""
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.sinc(2 * cutoff_hz / fs * n) * np.blackman(ntaps)
    return (h / h.sum()).astype(np.float32)


def fftfilt(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution via FFT, trimmed to len(x) with zero group delay
    (h is assumed symmetric / linear phase)."""
    n = len(x) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.ifft(np.fft.fft(x, nfft) * np.fft.fft(h, nfft))[:n]
    d = (len(h) - 1) // 2
    y = y[d:d + len(x)]
    return y if np.iscomplexobj(x) or np.iscomplexobj(h) else y.real


def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine pulse, `span` symbols long, unit energy."""
    t = (np.arange(span * sps + 1) - span * sps / 2) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta and abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) \
                / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def awgn(iq: np.ndarray, snr_db: float, ref_power: float, rng) -> np.ndarray:
    """Add complex white noise so that ref_power / noise_power = snr_db
    (noise measured over the full sample-rate bandwidth)."""
    npow = ref_power / 10 ** (snr_db / 10)
    noise = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))) * np.sqrt(npow / 2)
    return (iq + noise).astype(np.complex64)


def fm_modulate(msg: np.ndarray, dev_hz: float, fs: float) -> np.ndarray:
    """Continuous-phase FM: msg in [-1, 1] -> exp(j*2*pi*dev*integral(msg))."""
    phase = 2 * np.pi * dev_hz * np.cumsum(msg) / fs
    return np.exp(1j * phase).astype(np.complex64)


def fm_demod(iq: np.ndarray, fs: float) -> np.ndarray:
    """Quadrature discriminator, output in Hz."""
    d = np.angle(iq[1:] * np.conj(iq[:-1])) * fs / (2 * np.pi)
    return np.concatenate([d[:1], d]).astype(np.float32)


# --------------------------------------------------------------------------
# Bits and CRCs
# --------------------------------------------------------------------------

def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    return np.packbits(bits[: len(bits) // 8 * 8]).tobytes()


def crc16_ccitt_false(data: bytes) -> int:
    """poly 0x1021, init 0xFFFF, no reflection, no xorout (same as demod_chal)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def find_bits(bits: np.ndarray, pattern: np.ndarray, max_errors: int = 0) -> list[int]:
    """Indices where `pattern` occurs in `bits` with <= max_errors mismatches."""
    bits = np.asarray(bits, dtype=np.int8)
    pattern = np.asarray(pattern, dtype=np.int8)
    L = len(pattern)
    if len(bits) < L:
        return []
    win = np.lib.stride_tricks.sliding_window_view(bits, L)
    errs = np.sum(win != pattern, axis=1)
    return list(np.nonzero(errs <= max_errors)[0])
