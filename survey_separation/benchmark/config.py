"""Dataset discovery. The dataset lives OUTSIDE this tree; pass its root in."""
from __future__ import annotations
from pathlib import Path

from .core.separator import Track
from .core.stems import CANONICAL


def load_musdb_tracks(root, subset: str = "test"):
    """MUSDB18-HQ layout: <root>/<subset>/<track>/{mixture,vocals,drums,bass,other}.wav"""
    base = Path(root) / subset
    if not base.is_dir():
        raise FileNotFoundError(f"MUSDB subset not found: {base}")
    tracks = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        mixture = d / "mixture.wav"
        refs = {s: d / f"{s}.wav" for s in CANONICAL}
        if mixture.exists() and all(r.exists() for r in refs.values()):
            tracks.append(Track(id=d.name, mixture=mixture, refs=refs))
    return tracks
