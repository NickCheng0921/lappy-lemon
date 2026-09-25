"""Bases for tools whose stems arrive as files rather than via an API call.

InboxSeparator   - human renders in a GUI and drops WAVs in <dir>/inbox/<id>/
CaptureSeparator - real-time output recorded to <dir>/captures/<id>/ (aligned
                   downstream by the evaluator's coarse_align)
Both raise StemsPending until the expected files exist.
"""
from __future__ import annotations
from pathlib import Path

from benchmark.core.separator import Separator, StemsPending
from benchmark.core.stems import StemSet


class _FolderSeparator(Separator):
    subdir = "inbox"
    ingest = "manual"

    def __init__(self, name: str, produces, root):
        self.name = name
        self.produces = set(produces)
        self.root = Path(root)

    def resolve(self, track, workdir):
        src = self.root / self.subdir / track.id
        missing = [s for s in self.produces if not (src / f"{s}.wav").exists()]
        if missing:
            raise StemsPending(self.name, track.id, missing)
        paths = {s: src / f"{s}.wav" for s in self.produces}
        return StemSet(track.id, paths, meta={"tool": self.name, "ingest": self.ingest})


class InboxSeparator(_FolderSeparator):
    subdir = "inbox"
    ingest = "manual"


class CaptureSeparator(_FolderSeparator):
    subdir = "captures"
    ingest = "capture"
