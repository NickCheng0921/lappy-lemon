"""Tool-agnostic evaluation: StemSet + reference -> per-stem BSS Eval v4 rows.

Uses museval.metrics.bss_eval directly (images version, whole-signal filter),
so ingestion source is irrelevant. Per-track value is the median over windows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .audio import load_stereo, coarse_align, match_length
from .stems import PROFILES, NONVOCAL


def _profile_arrays(paths: dict, profile: str, sr: int):
    """Load (T, ch) arrays for each stem in `profile`, summing accompaniment."""
    out = {}
    for s in PROFILES[profile]:
        if s in paths:
            out[s] = load_stereo(paths[s], sr)
        elif s == "accompaniment":
            comps = [load_stereo(paths[c], sr) for c in NONVOCAL if c in paths]
            if not comps:
                return None
            t = min(len(c) for c in comps)
            out[s] = sum(c[:t] for c in comps)
        else:
            return None
    return out


def evaluate_track(track, stemset, profile, separator_name,
                   sr: int = 44100, win_s: float = 1.0) -> pd.DataFrame:
    import museval

    ref = _profile_arrays(track.refs, profile, sr)
    est = _profile_arrays(stemset.paths, profile, sr)
    if ref is None or est is None:
        return pd.DataFrame()

    stems = PROFILES[profile]
    R, E = [], []
    for s in stems:
        r, e = ref[s], est[s]
        e = coarse_align(r, e, sr)
        r, e = match_length(r, e)
        R.append(r)
        E.append(e)
    t = min(x.shape[0] for x in R + E)
    R = np.stack([x[:t] for x in R])       # (nsrc, T, ch)
    E = np.stack([x[:t] for x in E])

    win = int(win_s * sr)
    try:
        sdr, isr, sir, sar, _ = museval.metrics.bss_eval(
            R, E, window=win, hop=win,
            framewise_filters=False,        # one filter over the whole signal (v4)
            bsseval_sources_version=False,  # images version (MUSDB convention)
            compute_permutation=False,
        )
        status = "ok"
    except Exception as exc:                # e.g. a silent stem
        z = np.full((len(stems), 1), np.nan)
        sdr = isr = sir = sar = z
        status = f"error:{type(exc).__name__}"

    rows = [dict(
        separator=separator_name, track_id=track.id, profile=profile, stem=s,
        sdr=float(np.nanmedian(sdr[i])), sir=float(np.nanmedian(sir[i])),
        sar=float(np.nanmedian(sar[i])), isr=float(np.nanmedian(isr[i])),
        status=status,
    ) for i, s in enumerate(stems)]
    return pd.DataFrame(rows)
