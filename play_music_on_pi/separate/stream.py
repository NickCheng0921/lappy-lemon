"""Sliding-window streaming with tail emission."""

import queue
import threading
import time

import numpy as np

from gains import GAIN_MAX, GAIN_MIN, SOURCES, pot_gain
from util import log


class Stream:
    """Sliding-window streaming with TAIL emission.

    The window is 7.8s of context, but all of it is lookback: we keep a rolling
    buffer of the most recent `win` samples and emit the freshly-computed TAIL
    each cycle. Nothing waits for a window to "fill", so the window length costs
    zero latency -- only `proc + stride + lookahead + preroll` remain.

    Two details that make it work:
      * lookahead -- the last frames of a window have no future context and
        measure ~7dB worse than centre. We stop short of the true edge by
        `lookahead` seconds and give up that much latency to get the quality back.
      * crossfade -- consecutive emissions come from *different* model runs, so
        they meet at a seam. We compute a little extra and crossfade the join.
    """

    def __init__(
        self, sep, gains, stride_s, lookahead_s, xfade_s, preroll_s, max_lag_s=None
    ):
        self.sep = sep
        self.win = sep.win
        sr = sep.sr
        self.stride = int(stride_s * sr)
        self.look = int(lookahead_s * sr)
        self.xfade = int(xfade_s * sr)
        self.preroll_s = preroll_s

        if self.stride + self.look + self.xfade > self.win:
            raise SystemExit(
                "stride + lookahead + xfade must fit inside the "
                f"{self.win/sr:.2f}s window"
            )

        self.gain_map = dict(gains)
        self.master = 1.0  # post-model level control
        self._rebuild_gains()

        # Absolute-position buffer. A fixed rolling window has a FLOATING right
        # edge: if a feed lands between two worker passes, the next window's
        # edge advances by more than `stride` and the emitted region silently
        # skips audio. Tracking absolute sample indices makes each window's
        # position exact regardless of feed granularity.
        #   buf holds absolute samples [self.base, self.base + buf.shape[1])
        #   next_end = absolute right edge of the window to process next
        self.base = -(self.win - self.stride)
        self.inbuf = np.zeros((sep.ch, self.win - self.stride), dtype=np.float32)
        self.next_end = self.stride
        self.incond = threading.Condition()
        self.outq = queue.Queue(maxsize=8)  # blocks hold 4 stems each
        self.done = threading.Event()

        self.held = None  # xfade tail from the last block
        f = np.linspace(0.0, 1.0, self.xfade, dtype=np.float32)[None, :]
        self.fade_in, self.fade_out = f, 1.0 - f

        # Backlog bound. Without it a worker that ever runs slower than
        # realtime falls further behind every window and never recovers, since
        # next_end only ever advances by one stride. None = never drop (offline).
        self.max_lag = None if max_lag_s is None else int(max_lag_s * sep.sr)
        self.dropped = 0

        self.processed = 0
        self.t_proc = 0.0

    # ----------------------------------------------------------------- mix
    def _rebuild_gains(self):
        # Replacing the whole array is an atomic rebind, so the worker either
        # sees the old gains or the new ones -- never a half-updated mix.
        self.gains = (
            np.array([self.gain_map[s] for s in SOURCES], dtype=np.float32)
            * self.master
        )[:, None, None]

    def mixdown(self, stems_block):
        """[4, ch, n] -> [ch, n] using the CURRENT gains. Called per output
        chunk by the writer, which is what makes the keys feel instant."""
        return (stems_block * self.gains).sum(0)

    def set_positions(self, positions):
        """Absolute fader positions (0..1, the stems then master) -> gains.

        One _rebuild_gains() for the whole set, not one per fader: the rebuild
        is an atomic rebind, so the worker sees either the old mix or the new
        one and never new vocals against an old master.
        """
        for name, pos in zip(SOURCES, positions):
            self.gain_map[name] = pot_gain(pos)
        self.master = pot_gain(positions[len(SOURCES)])
        self._rebuild_gains()
        return self.gain_map

    # ---------------------------------------------------------------- input
    def feed(self, block):
        """Append new samples. Positions are absolute, so feed granularity is
        irrelevant -- partial chunks can no longer shift a window's edge."""
        with self.incond:
            self.inbuf = np.concatenate([self.inbuf, block], axis=1)
            self.incond.notify_all()

    def finish_input(self):
        self.done.set()
        with self.incond:
            self.incond.notify_all()

    # --------------------------------------------------------------- worker
    def worker(self):
        while True:
            with self.incond:
                have = lambda: self.base + self.inbuf.shape[1] >= self.next_end
                while not have() and not self.done.is_set():
                    self.incond.wait(timeout=0.5)
                if not have():
                    break  # input ended mid-window
                newest = self.base + self.inbuf.shape[1]
                if self.max_lag is not None and newest - self.next_end > self.max_lag:
                    # Too far behind: jump to the newest complete window and
                    # discard the backlog. Drops audio, but bounded latency is
                    # the point of a live monitor.
                    skip = newest - self.next_end
                    self.dropped += skip
                    self.next_end = newest
                    log(
                        "behind -- skipped %.2fs to stay current (%.1fs total)"
                        % (skip / self.sep.sr, self.dropped / self.sep.sr)
                    )
                end = self.next_end
                st = end - self.win - self.base
                window = self.inbuf[:, st : st + self.win].copy()
                self.next_end += self.stride
                # drop what no future window can reference
                keep = max(0, (self.next_end - self.win) - self.base)
                if keep:
                    self.inbuf = self.inbuf[:, keep:]
                    self.base += keep
                self.incond.notify_all()

            t0 = time.perf_counter()
            stems = self.sep.separate(window)  # [4, ch, win]
            self.t_proc += time.perf_counter() - t0
            self.processed += 1

            # Tail slice, stopping `look` short of the raw edge. Length is
            # stride+xfade so the extra tail can be blended into the next block.
            # Queue the STEMS, not a mix. Gains are applied by the writer at
            # playback time, so a keypress changes what you hear on the next
            # output chunk (~50ms) instead of waiting a full pipeline delay
            # for freshly-separated audio to arrive.
            end = self.win - self.look
            seg = stems[:, :, end - self.stride - self.xfade : end].copy()

            if self.held is not None and self.xfade:
                # same absolute time range as the previous block's held tail.
                # Crossfading per-stem then summing == crossfading the sum.
                seg[:, :, : self.xfade] = (
                    seg[:, :, : self.xfade] * self.fade_in + self.held * self.fade_out
                )
            self.held = seg[:, :, self.stride :].copy() if self.xfade else None
            self.outq.put(seg[:, :, : self.stride].copy())

            if self.processed % 5 == 0:
                rtf = self.t_proc / (self.processed * self.stride / self.sep.sr)
                warn = "  !! CANNOT KEEP UP" if rtf > 1 else ""
                log(f"  blocks={self.processed}  proc/stride={rtf:.2f}x{warn}")
        self.outq.put(None)
