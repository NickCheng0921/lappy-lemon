"""Rendering helpers for the summary table."""
from __future__ import annotations
import pandas as pd


def markdown_table(summary: pd.DataFrame, profile: str = "4stem",
                   metric: str = "sdr") -> str:
    s = summary[summary.profile == profile]
    if s.empty:
        return "(no results for this profile)"
    piv = s.pivot(index="separator", columns="stem", values=metric)
    try:
        return piv.round(2).to_markdown()
    except Exception:               # tabulate not installed
        return piv.round(2).to_string()
