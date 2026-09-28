"""Shared base for Demucs CLI models (htdemucs, htdemucs_ft, htdemucs_6s...)."""
from __future__ import annotations
import subprocess
from pathlib import Path

from benchmark.core.separator import Separator
from benchmark.core.stems import StemSet, CANONICAL


class DemucsSeparator(Separator):
    produces = set(CANONICAL)

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    def resolve(self, track, workdir):
        # Per-track out_root: every MUSDB mixture file is named "mixture.wav", so
        # keying on track.mixture.stem would collide across tracks and make the
        # cache check reuse the first track's stems for all of them.
        out_root = Path(workdir) / self.name / track.id
        # demucs writes <out_root>/<model>/<mixture_stem>/{stem}.wav
        stem_dir = out_root / self.model / track.mixture.stem
        if not all((stem_dir / f"{s}.wav").exists() for s in CANONICAL):
            out_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["demucs", "-n", self.model, "-o", str(out_root), str(track.mixture)],
                check=True,
            )
        paths = {s: stem_dir / f"{s}.wav" for s in CANONICAL}
        return StemSet(track.id, paths, meta={"tool": "demucs", "model": self.model})
