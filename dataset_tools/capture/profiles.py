"""Per-class SDR++ display + generator profiles (randomized per capture session).

rtl_sr / decimation set the visible span; fft_rate sets how many seconds one screen covers
(slow modes get low rates so their structure is visible, bursty modes get high rates).
gen_fs is the generator sample rate (must exceed the signal bandwidth).
"""
import numpy as np

RTL_RATES = [250_000, 1_024_000, 2_048_000, 2_400_000]
# Measured on this Mac (square-wave scroll test): SDR++ draws every line while FFT size x rate <= ~50M bins/s;
# above that it drops lines unevenly and the time axis becomes irregular.
REALTIME_BINS_PER_S = 50e6

# Wideband classes need more TX power for the same per-bin SNR
TX_GAIN = {c: (14, 30) for c in ["BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "fm", "lora", "z-wave", "ads-b", "cellular",
                                 "digital-audio-broadcasting", "bluetooth", "2ASK", "OOK",
                                 "remote-keyless-entry"]}

# Share of sessions using the big-FFT / fast-waterfall SDR++ look (default 0.4)
HIRES_P = {"2FSK": 0.7}

# class: (rtl_sr choices, decimation choices, fft_rate range, fft_size choices, gen_fs)
PROFILES = {
    "morse":                          ([250_000], [8, 16, 32], (40, 120), [16384, 32768], 250e3),
    "OOK":                  ([1_024_000, 250_000], [1], (200, 900), [1024, 2048, 4096], 1e6),
    "2ASK":                           ([250_000, 1_024_000], [1, 2, 4], (150, 600), [8192, 16384, 32768], 1e6),
    "2FSK":                           ([2_400_000, 1_024_000, 250_000], [1, 2, 4], (300, 1500), [8192, 16384, 32768], 1e6),
    "4FSK":                           ([250_000], [2, 4, 8], (100, 400), [16384, 32768], 250e3),
    "BPSK":                           ([1_024_000, 2_400_000], [1], (150, 600), [8192, 16384], 2.4e6),
    "QPSK":                           ([1_024_000, 2_400_000], [1], (150, 600), [8192, 16384], 2.4e6),
    "8PSK":                           ([1_024_000, 2_400_000], [1], (150, 600), [8192, 16384], 2.4e6),
    "16QAM":                          ([1_024_000, 2_400_000], [1], (150, 600), [8192, 16384], 2.4e6),
    "32QAM":                          ([1_024_000, 2_400_000], [1], (150, 600), [8192, 16384], 2.4e6),
    "am":                             ([250_000], [4, 8], (60, 200), [16384, 32768], 250e3),
    "airband":                        ([250_000], [4, 8], (40, 120), [16384, 32768], 250e3),
    "fm":                             ([1_024_000, 2_400_000], [1], (60, 250), [16384, 32768], 1e6),
    "pocsag":                         ([250_000], [4, 8], (100, 300), [16384, 32768], 250e3),
    "ads-b":                          ([2_400_000, 2_048_000], [1], (1200, 3000), [1024], 2.4e6),
    "ais":                            ([250_000], [1, 2], (150, 500), [8192, 16384], 250e3),
    "packet":                         ([250_000], [4, 8], (60, 200), [16384, 32768], 250e3),
    "Radioteletype":                  ([250_000], [32, 64], (30, 100), [16384, 32768], 250e3),
    "sstv":                           ([250_000], [8, 16], (40, 150), [16384, 32768], 250e3),
    "automatic-picture-transmission": ([250_000], [2, 4], (40, 150), [16384, 32768], 250e3),
    "lora":                           ([1_024_000, 2_400_000], [1], (1000, 3000), [1024, 2048, 4096], 2.4e6),
    "RS41-Radiosonde":                ([250_000], [4, 8, 16], (80, 250), [16384, 32768], 250e3),
    "remote-keyless-entry":           ([1_024_000, 250_000], [1], (200, 800), [1024, 2048], 1e6),
    "z-wave":                         ([1_024_000, 2_400_000], [1], (300, 1200), [1024, 2048], 2.4e6),
    "digital-speech-decoder":         ([250_000], [4, 8, 16], (80, 400), [16384, 32768], 250e3),
    "vor":                            ([250_000], [4, 8], (60, 200), [16384, 32768], 250e3),
    "digital-audio-broadcasting":     ([2_048_000, 2_400_000], [1], (60, 400), [8192, 16384, 32768], 2.4e6),
    "cellular":                       ([2_400_000, 2_048_000], [1], (150, 800), [8192, 16384], 2.4e6),
    "bluetooth":                      ([2_400_000], [1], (800, 2500), [1024, 2048], 2.4e6),
}


def pick(cls: str, rng, hires: bool | None = None) -> dict:
    srs, decs, (r0, r1), sizes, gen_fs = PROFILES[cls]
    sr = int(rng.choice(srs))
    dec = int(rng.choice(decs))
    if sr > 300_000 and dec > 2:
        dec = 2
    hires = bool(rng.random() < HIRES_P.get(cls, 0.4)) if hires is None else hires  # "hi-res" SDR++ style: big FFT, fast waterfall (like 32k/65k @ 1-3k fps)
    if rng.random() < 0.15:   # as users set it, beyond what this Mac renders in real time (irregular time axis)
        hr_size, hr_rate = int(rng.choice([32768, 65536])), int(rng.integers(1000, 3001))
    else:                     # real-time combinations (measured: FFT size x rate <= ~50M bins/s)
        hr_size, (lo_r, hi_r) = [(65536, (500, 750)), (32768, (1000, 1500)), (16384, (2000, 3000))][int(rng.integers(3))]
        hr_rate = int(rng.integers(lo_r, hi_r + 1))
    rtl_gain = int(rng.choice([10, 15, 20, 25, 30]))
    floor = -84 + (rtl_gain - 20) * 0.6         # noise floor shifts with RTL gain
    return {
        "rtl_sr": sr,
        "decimation": dec,
        "fft_rate": hr_rate if hires else int(rng.integers(r0, r1 + 1)),
        "fft_size": hr_size if hires else int(rng.choice(sizes)),
        "hires": hires,
        "overloaded": hires and hr_size * hr_rate > REALTIME_BINS_PER_S,
        "gen_fs": gen_fs,
        "rtl_gain": rtl_gain,
        "min_db": round(float(floor), 1),      # refined by run_capture.autolevel
        "range_db": round(float(rng.uniform(40, 60)), 1),
        "max_db": None,
        "tx_gain": TX_GAIN.get(cls, (0, 22)),
    }
