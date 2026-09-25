"""Separator contract. One method: resolve(track) -> StemSet.

There is no explicit mode. Behaviour differs by adapter:
  - generating adapters (Demucs, Spleeter) shell out and return paths;
  - inbox/capture adapters read human-rendered or recorded files and raise
    StemsPending when they aren't there yet.
The runner treats them identically and just catches StemsPending.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Track:
    id: str
    mixture: Path
    refs: dict                # canonical stem -> Path (ground truth)


class StemsPending(Exception):
    """Raised by manual/capture adapters when stems haven't been produced yet."""
    def __init__(self, separator: str, track_id: str, missing):
        self.separator = separator
        self.track_id = track_id
        self.missing = list(missing)
        super().__init__(f"[{separator}] {track_id}: awaiting {self.missing}")


class Separator:
    name: str = "base"
    produces: set = set()     # canonical stem names this tool can emit

    def resolve(self, track: "Track", workdir: Path) -> "StemSet":
        raise NotImplementedError
