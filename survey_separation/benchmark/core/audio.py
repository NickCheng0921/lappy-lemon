"""Audio IO and pre-eval conditioning: load, stereo, coarse alignment, trim.

museval requires reference and estimate to be identical shape. Everything is
forced to a common sample rate + stereo, coarsely delay-aligned (BSS Eval's
512-tap filter only forgives ~11.6 ms, so capture/GUI latency must be removed
first), then truncated to a common length.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


def load_stereo(path, sr: int) -> np.ndarray:
    """Load `path` as float32 (T, 2) at sample rate `sr`."""
    try:
        import librosa
        y, _ = librosa.load(str(path), sr=sr, mono=False)   # (ch, T) or (T,)
        x = y.T if y.ndim == 2 else np.stack([y, y], axis=1)
    except Exception:
        import soundfile as sf
        x, file_sr = sf.read(str(path), always_2d=True)     # (T, ch)
        if file_sr != sr:
            raise RuntimeError(
                f"{path}: {file_sr} Hz != target {sr} Hz and librosa unavailable "
                f"for resampling")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return np.ascontiguousarray(x[:, :2], dtype=np.float32)


def match_length(a: np.ndarray, b: np.ndarray):
    t = min(a.shape[0], b.shape[0])
    return a[:t], b[:t]


def coarse_align(ref: np.ndarray, est: np.ndarray, sr: int,
                 max_lag_s: float = 1.0) -> np.ndarray:
    """Integer-sample shift of `est` to best match `ref` (GCC on mono mixdown).

    Coarse on purpose: it removes gross capture/export latency; BSS Eval's own
    filter mops up the sub-frame residual. Near no-op for sample-accurate tools.
    """
    from scipy.signal import fftconvolve
    r = ref.mean(axis=1)
    e = est.mean(axis=1)
    n = min(len(r), len(e))
    if n < sr // 4:                       # too short to align reliably
        return est
    r, e = r[:n], e[:n]
    corr = fftconvolve(e, r[::-1], mode="full")
    lags = np.arange(-n + 1, n)
    max_lag = int(max_lag_s * sr)
    mid = n - 1
    lo, hi = max(0, mid - max_lag), min(len(corr), mid + max_lag + 1)
    lag = int(lags[lo:hi][int(np.argmax(corr[lo:hi]))])
    if lag > 0:
        est = est[lag:]
    elif lag < 0:
        est = np.pad(est, ((-lag, 0), (0, 0)))
    return est
