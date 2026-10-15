# SPDX-License-Identifier: GPL-3.0-or-later
"""Loudness measurement and normalization with NumPy only (no FFmpeg, no SciPy).

Integrated loudness follows ITU-R BS.1770-4 / EBU R 128 (K-weighting at any sample rate, 400 ms blocks, absolute and
relative gates); true peak uses 4x oversampling. ``normalize`` sets the integrated loudness and keeps the true peak at
or below the ceiling with a look-ahead limiter, repeating while limiting lowers the loudness.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _k_weighting_response(n_fft: int, rate: int) -> np.ndarray:
    """Frequency response of the BS.1770 K-weighting filter (shelf, then RLB high-pass) designed for ``rate``."""
    def biquad_response(b, a, w):
        z = np.exp(-1j * w)
        return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    # Shelf (libebur128 / BS.1770 design at any rate).
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh, vb = 10 ** (gain_db / 20), (10 ** (gain_db / 20)) ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf_b = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    shelf_a = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    # RLB high-pass.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1 + k / q + k * k
    hp_b = [1.0, -2.0, 1.0]
    hp_a = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    w = 2 * np.pi * np.fft.rfftfreq(n_fft) # radians per sample
    return biquad_response(shelf_b, shelf_a, w) * biquad_response(hp_b, hp_a, w)


def _k_weighted(x: np.ndarray, rate: int) -> np.ndarray:
    pad = int(rate)  # the filter's impulse response has died away within a second
    n = 1 << int(math.ceil(math.log2(x.size + pad)))
    spectrum = np.fft.rfft(x.astype(np.float64), n) * _k_weighting_response(n, rate)
    return np.fft.irfft(spectrum, n)[: x.size]


def _block_powers(y: np.ndarray, rate: int, seconds: float, hop: float) -> np.ndarray:
    size, step = int(round(seconds * rate)), int(round(hop * rate))
    if y.size < size:
        return np.empty(0)
    c = np.concatenate([[0.0], np.cumsum(y * y)])
    starts = np.arange(0, y.size - size + 1, step)
    return (c[starts + size] - c[starts]) / size


def _lufs(z):
    with np.errstate(divide="ignore"):
        return -0.691 + 10 * np.log10(z)


def integrated_loudness(samples: np.ndarray, rate: int) -> Optional[float]:
    z = _block_powers(_k_weighted(samples, rate), rate, 0.4, 0.1)
    z = z[_lufs(z) > -70]
    if not z.size:
        return None
    z = z[_lufs(z) > _lufs(z.mean()) - 10]
    return float(_lufs(z.mean())) if z.size else None


# 4x interpolator, 12 taps per phase (48 taps), windowed sinc as in BS.1770-4 Annex 2.
_PHASES, _TAPS = 4, 12


def _kernel() -> np.ndarray:
    n = _PHASES * _TAPS
    i = np.arange(n)
    x = (i - (n - 1) / 2) / _PHASES
    w = 0.42 - 0.5 * np.cos(2 * np.pi * i / (n - 1)) + 0.08 * np.cos(4 * np.pi * i / (n - 1))
    proto = np.sinc(x) * w
    rows = np.stack([proto[p::_PHASES] for p in range(_PHASES)])
    return rows / rows.sum(axis=1, keepdims=True)


_KERNEL = _kernel()


def true_peak(samples: np.ndarray) -> float:
    """Highest absolute value after 4x oversampling (linear)."""
    x = samples.astype(np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    for row in _KERNEL:
        if x.size >= _TAPS:
            peak = max(peak, float(np.max(np.abs(np.convolve(x, row, mode="valid")))))
    return peak


def db(value: float) -> Optional[float]:
    return 20 * math.log10(value) if value > 0 else None


def measure(samples: np.ndarray, rate: int) -> dict:
    tp = true_peak(samples)
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
    return {"integrated_lufs": integrated_loudness(samples, rate), "true_peak_dbtp": db(tp), "rms_dbfs": db(rms),
            "duration_seconds": samples.size / rate}


def limit(samples: np.ndarray, rate: int, ceiling_db: float) -> np.ndarray:
    """Look-ahead peak limiter: the gain is a moving average of a two-sided minimum hold of the gain each sample
    needs, so it reaches full reduction before every peak and recovers within about 10 ms."""
    w = max(1, int(round(0.005 * rate)))
    c = 10 ** (ceiling_db / 20)
    a = np.abs(samples.astype(np.float64))
    need = np.where(a > c, c / np.maximum(a, 1e-12), 1.0)
    padded = np.pad(need, w, constant_values=1.0)
    held = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1).min(axis=1)
    cs = np.concatenate([[0.0], np.cumsum(np.pad(held, w, mode="edge"))])
    smooth = (cs[2 * w + 1:] - cs[: -(2 * w + 1)]) / (2 * w + 1)
    return (samples * smooth[: samples.size]).astype(np.float32)


def normalize(samples: np.ndarray, rate: int, target_lufs: float = -16.0, true_peak_db: float = -1.5) -> tuple:
    """Audio at ``target_lufs`` with true peak at or below ``true_peak_db``; returns (audio, report)."""
    loud = integrated_loudness(samples, rate)
    if loud is None:
        return samples, {"normalized": False, "limited": False}
    gain_db, ceiling, limited = target_lufs - loud, true_peak_db - 0.5, False
    out = samples
    for _ in range(6):
        out = (samples * 10 ** (gain_db / 20)).astype(np.float32)
        tp = db(true_peak(out))
        if tp is not None and tp > true_peak_db:
            out = limit(out, rate, ceiling)
            limited = True
        now = integrated_loudness(out, rate)
        tp = db(true_peak(out))
        off = target_lufs - now if now is not None else 0.0
        if abs(off) <= 0.1 and (tp is None or tp <= true_peak_db + 0.05):
            break
        if abs(off) <= 0.1:
            ceiling -= (tp - true_peak_db) + 0.1
        gain_db += off
    return out, {"normalized": True, "limited": limited}
