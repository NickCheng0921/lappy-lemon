"""Canonical stem taxonomy and comparison profiles.

Every adapter, whatever tool it wraps, reports stems using these canonical
names. Tools that don't produce the full 4-stem set (DJ tools, the hardware
box) declare a smaller `produces` set and are only scored on profiles they
can fill.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

CANONICAL = ("vocals", "drums", "bass", "other")
NONVOCAL = ("drums", "bass", "other")

# A profile is the set of stems compared in one evaluation pass.
#   4stem: the classic MUSDB split (Demucs / Spleeter / FL Studio)
#   2stem: lowest common denominator — everything can be scored here,
#          including 2-stem DJ tools and the hardware box.
PROFILES = {
    "4stem": ["vocals", "drums", "bass", "other"],
    "2stem": ["vocals", "accompaniment"],
}


@dataclass
class StemSet:
    """The unified currency of the harness: named stem WAVs for one track."""
    track_id: str
    paths: dict               # stem name -> Path
    meta: dict = field(default_factory=dict)

    def available(self) -> set:
        return set(self.paths)


def can_fill(produces, profile: str) -> bool:
    """True if a tool emitting `produces` can supply every stem in `profile`.

    `accompaniment` is satisfied either natively or by summing drums+bass+other.
    """
    prod = set(produces)
    for s in PROFILES[profile]:
        if s in prod:
            continue
        if s == "accompaniment" and set(NONVOCAL).issubset(prod):
            continue
        return False
    return True
