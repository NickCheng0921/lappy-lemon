"""VirtualDJ - MANUAL. Exported as the 4 canonical stems, so it is scored on
both the 4stem and 2stem profiles (accompaniment is summed from drums+bass+other
downstream).

Drop exports at:
  separators/virtualdj/inbox/<track_id>/{vocals,drums,bass,other}.wav

The .vdjstems cache is just an MP4 holding 5 audio streams (1-vocal, 2-hihat,
3-bass, 4-instruments, 5-kick), same trick as Traktor STEMS. Rename to .mp4 or
demux with ffmpeg (`ffmpeg -i x.vdjstems -map 0:0 vocals.wav ...`); order per
ffmpeg -i. Community CLI wrapper: https://github.com/i0x0/vdjstems_cli

Map VDJ's 5 streams onto the canonical 4:
  vocals = vocal (1)
  bass   = bass (3)
  drums  = kick (5) + hihat (2)     # sum the two percussive streams
  other  = instruments (4)
For live/real-time capture instead, use CaptureSeparator (captures/) and let the
evaluator's coarse_align remove the latency.
"""
from pathlib import Path

from separators._file_bases import InboxSeparator
from benchmark.core.stems import CANONICAL

HERE = Path(__file__).parent


def build():
    return InboxSeparator("virtualdj", set(CANONICAL), HERE)
