"""FL Studio AI stem separation - MANUAL (GUI-only).

Workflow: right-click the mixture clip -> stem separation -> export each stem.
Drop the WAVs at:  separators/flstudio/inbox/<track_id>/{vocals,drums,bass,other}.wav
(<track_id> is the MUSDB track folder name.) Until then the runner logs it in
pending.json.
"""
from pathlib import Path

from separators._file_bases import InboxSeparator
from benchmark.core.stems import CANONICAL

HERE = Path(__file__).parent


def build():
    return InboxSeparator("flstudio", set(CANONICAL), HERE)
