"""Spleeter 4stems (Deezer) - AUTO via CLI.

Deprecated upstream since 2022 but still installs/runs. TensorFlow deps clash
with torch/demucs, so run it in its own venv and pass that interpreter as
spleeter_python (defaults to whatever `python` resolves to).
"""
from __future__ import annotations
import subprocess
from pathlib import Path

from benchmark.core.separator import Separator
from benchmark.core.stems import StemSet, CANONICAL


class SpleeterSeparator(Separator):
    name = "spleeter"
    produces = set(CANONICAL)

    def __init__(self, python_bin: str = "python"):
        self.python = python_bin

    def resolve(self, track, workdir):
        out_root = Path(workdir) / self.name
        # spleeter writes <out_root>/<mixture_stem>/{stem}.wav
        stem_dir = out_root / track.mixture.stem
        if not all((stem_dir / f"{s}.wav").exists() for s in CANONICAL):
            out_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [self.python, "-m", "spleeter", "separate",
                 "-p", "spleeter:4stems", "-o", str(out_root), str(track.mixture)],
                check=True,
            )
        paths = {s: stem_dir / f"{s}.wav" for s in CANONICAL}
        return StemSet(track.id, paths, meta={"tool": "spleeter", "model": "4stems"})


def build(python_bin: str = "python"):
    return SpleeterSeparator(python_bin)
