"""Audio in and out: ffmpeg decode, int16 conversion, pipe sizing."""

import subprocess

import numpy as np

from util import log


def to_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")

def read_audio(path, sr, ch):
    """Decode any input to float32 [ch, n] via ffmpeg.

    Deliberately NOT demucs.audio.AudioFile: that module imports lameenc at
    import time (for mp3 *writing*, which we never do), so it drags an extra
    native dependency onto the Pi for no benefit. ffmpeg is already required.
    """
    cmd = [
        "ffmpeg",
        "-v",
        "quiet",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(ch),
        "-",
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    a = np.frombuffer(raw, dtype="<f4").reshape(-1, ch).T
    return np.ascontiguousarray(a)

def _cap_pipe(fobj, nbytes, sep):
    """Shrink the pipe feeding aplay. Returns its size in ms of audio.

    Best-effort: a kernel without F_SETPIPE_SZ, or one that refuses the size,
    costs latency but nothing else, so warn and carry on rather than refusing
    to play. The kernel rounds to a page and reports what it actually did.
    """
    import fcntl

    def as_ms(n):
        return n / (sep.sr * sep.ch * 2) * 1000.0

    fd = fobj.fileno()
    try:
        before = fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ)
        fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, nbytes)
        after = fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ)
    except (AttributeError, OSError) as e:
        log(
            f"warning: could not resize the aplay pipe ({e}); "
            "key response will lag by however much it holds"
        )
        return None
    log(
        f"aplay pipe {before} -> {after} bytes ({as_ms(before):.0f}ms -> "
        f"{as_ms(after):.0f}ms of audio held at the old gain)"
    )
    return as_ms(after)
