"""VirtualDJ - MANUAL. DJ-oriented stems, so treated as 2-stem
(vocals + accompaniment); only the 2stem profile applies.

Drop exports at:
  separators/virtualdj/inbox/<track_id>/{vocals,accompaniment}.wav
For live/real-time capture instead, use CaptureSeparator (captures/) and let the
evaluator's coarse_align remove the latency.
"""
from pathlib import Path

from separators._file_bases import InboxSeparator

HERE = Path(__file__).parent


def build():
    return InboxSeparator("virtualdj", {"vocals", "accompaniment"}, HERE)
