"""In-place fader display for separate_stream.py.

Kept in its own module for the same reason startup_bar() is hand-rolled: this
is cosmetic, and cosmetic code must never be able to take the audio down. The
caller treats a missing or broken display as "no display", never as an error.

One row per fader, redrawn over itself:

     drums |################:######     |  +3.0dB
      bass |###########     :           |  -4.5dB
    vocals |                :           |   mute
                            ^ unity tick, POT_UNITY_POS of the travel

Sub-character resolution comes from the Unicode block-element run, the same
trick tqdm uses; a terminal whose encoding cannot carry those falls back to
plain '#'. Redraw is cursor-up + erase-line, so the block stays put instead of
scrolling.

Note on ssh: `ssh host "cmd"` allocates no tty, so isatty() is False on the Pi
even though the escapes would render fine on the laptop's terminal at the other
end of the pipe. That is what mode="ansi" is for -- force it when you know a
real terminal is reading.
"""

import sys
import time

ESC = chr(27)
CURSOR_UP = ESC + "[%dA"
ERASE_LINE = ESC + "[2K"

BLOCKS = " " + "".join(chr(0x258F - i) for i in range(7)) + chr(0x2588)
BLOCKS_ASCII = " #"


class FaderViz:
    """Multi-line fader block, redrawn in place.

    Draws are self-throttled: the fader thread polls at 20 Hz to keep the audio
    responsive, but the eye does not need 20 redraws a second and a terminal
    over ssh really does not.
    """

    def __init__(self, labels, out=None, width=28, mode="auto",
                 min_interval=0.1, unity_pos=None):
        self.labels = list(labels)
        self.out = out if out is not None else sys.stderr
        self.width = width
        self.min_interval = min_interval
        self.lw = max(len(x) for x in self.labels)

        self.blocks = BLOCKS
        if not self._encodable(BLOCKS):
            self.blocks = BLOCKS_ASCII

        if mode == "auto":
            mode = "ansi" if self._isatty() else "plain"
        self.mode = mode

        self.unity_col = None
        if unity_pos is not None:
            self.unity_col = min(int(unity_pos * width), width - 1)

        self._drawn = False
        self._last = 0.0

    # ------------------------------------------------------------ probing
    def _isatty(self):
        try:
            return self.out.isatty()
        except Exception:
            return False

    def _encodable(self, s):
        enc = getattr(self.out, "encoding", None)
        try:
            s.encode(enc or "ascii")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------ drawing
    def _bar(self, pos):
        pos = min(max(pos, 0.0), 1.0)
        steps = len(self.blocks) - 1
        filled = pos * self.width
        full = min(int(filled), self.width)
        cells = [self.blocks[-1]] * full
        if full < self.width:
            part = int((filled - full) * steps)
            cells.append(self.blocks[part])
        cells += [" "] * (self.width - len(cells))
        # The tick shows where unity sits, which matters here because unity is
        # at 80% of travel rather than the middle or the top. Only drawn where
        # the fill has not already passed it -- being above it is self-evident.
        if self.unity_col is not None and cells[self.unity_col] == " ":
            cells[self.unity_col] = ":"
        return "".join(cells[:self.width])

    def _row(self, label, pos, gain):
        if gain <= 0.0:
            db = "  mute"
        else:
            import math
            db = "%+5.1fdB" % (20.0 * math.log10(gain))
        return "%*s |%s| %s" % (self.lw, label, self._bar(pos), db)

    def draw(self, positions, gains, force=False):
        now = time.time()
        if not force and (now - self._last) < self.min_interval:
            return
        self._last = now
        rows = [self._row(l, p, g)
                for l, p, g in zip(self.labels, positions, gains)]
        try:
            if self.mode == "ansi":
                buf = []
                if self._drawn:
                    buf.append(CURSOR_UP % len(rows))
                for r in rows:
                    buf.append(ERASE_LINE + r + "\n")
                self.out.write("".join(buf))
            else:
                # No cursor control: one compact line, appended not redrawn.
                self.out.write("  ".join(
                    "%s %s" % (l, d.strip())
                    for l, d in zip(self.labels,
                                    [r.rsplit("|", 1)[1] for r in rows])
                ) + "\n")
            self.out.flush()
            self._drawn = True
        except Exception:
            # A broken pipe or a terminal that vanished is not worth a stack
            # trace from a thread whose only job is to look nice.
            self.mode = "off"

    def invalidate(self):
        """Something else wrote to the console; start a fresh block.

        Without this the next cursor-up would count lines that are no longer
        ours and eat whatever was logged.
        """
        self._drawn = False
