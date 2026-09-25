"""Orchestrate tracks x separators x profiles.

For each (separator, track): resolve() the stems. If a manual/capture adapter
raises StemsPending, record it in a worklist and move on. For resolved stems,
evaluate every profile the tool can fill. Writes per-track JSON, a combined
parquet, a summary CSV, and pending.json.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

from .separator import StemsPending
from .stems import can_fill
from .evaluate import evaluate_track
from .aggregate import aggregate


def _err_df(sep, track_id, msg):
    return pd.DataFrame([dict(separator=sep, track_id=track_id, profile="-", stem="-",
                              sdr=np.nan, sir=np.nan, sar=np.nan, isr=np.nan, status=msg)])


def run(separators, tracks, profiles, run_dir, sr=44100, win_s=1.0):
    run_dir = Path(run_dir)
    est_dir = run_dir / "estimates"
    met_dir = run_dir / "metrics"
    est_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    frames, pending = [], []
    for sep in separators:
        for track in tracks:
            try:
                stemset = sep.resolve(track, est_dir)
            except StemsPending as p:
                pending.append(dict(separator=p.separator, track_id=p.track_id,
                                    missing=p.missing))
                continue
            except Exception as exc:
                frames.append(_err_df(sep.name, track.id,
                                      f"resolve_error:{type(exc).__name__}"))
                continue

            for profile in profiles:
                if not can_fill(sep.produces, profile):
                    continue
                df = evaluate_track(track, stemset, profile, sep.name, sr, win_s)
                if not df.empty:
                    df.to_json(met_dir / f"{sep.name}__{track.id}__{profile}.json",
                               orient="records")
                    frames.append(df)

    per_track = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not per_track.empty:
        try:
            per_track.to_parquet(run_dir / "results.parquet")
        except Exception:
            per_track.to_csv(run_dir / "results.csv", index=False)
        aggregate(per_track).to_csv(run_dir / "summary.csv", index=False)
    if pending:
        (run_dir / "pending.json").write_text(json.dumps(pending, indent=2))

    return per_track, pending
