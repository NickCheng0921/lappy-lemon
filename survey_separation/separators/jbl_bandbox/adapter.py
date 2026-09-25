"""JBL bandbox (hardware) - CAPTURE. Record device output via loopback/line-in.

Drop recordings at:
  separators/jbl_bandbox/captures/<track_id>/{vocals,accompaniment}.wav
The evaluator applies coarse cross-correlation alignment before scoring, so
gross capture latency is fine; keep sample rate/format consistent.
"""
from pathlib import Path

from separators._file_bases import CaptureSeparator

HERE = Path(__file__).parent


def build():
    return CaptureSeparator("jbl_bandbox", {"vocals", "accompaniment"}, HERE)
