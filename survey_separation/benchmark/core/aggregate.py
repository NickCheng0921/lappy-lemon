"""Median over tracks -> one row per (separator, profile, stem)."""
from __future__ import annotations
import pandas as pd


def aggregate(per_track: pd.DataFrame) -> pd.DataFrame:
    if per_track.empty:
        return per_track
    ok = per_track[per_track.status == "ok"]
    if ok.empty:
        return pd.DataFrame()
    return (ok.groupby(["separator", "profile", "stem"])[["sdr", "sir", "sar", "isr"]]
              .median()
              .reset_index())
