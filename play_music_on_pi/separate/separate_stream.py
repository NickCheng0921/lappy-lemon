"""Real-time 4-stem separation on the Pi, through a delayed audio buffer.

Pipeline
--------
    AUX in --> [input ring] --> window --> STFT --> ONNX core --> iSTFT
                                                      |
                                        4 stems (drums/bass/other/vocals)
                                                      |
                                     per-stem gains --> sum --> overlap-add
                                                      |
                                            [output ring] --> speakers

Why there is a delay, and why it is not a bug: the model needs a whole 7.8s
window before it can separate anything, then needs time to run it. So the output
necessarily lags the input by about one window plus the processing time. The
delay buffer is what makes that lag smooth instead of glitchy.

Stem gains are adjustable live from the keyboard while it runs:

    u i o p   +5%     drums  bass  other  vocals
    j k l ;   -5%
    r         reset every stem to 100%
    q         quit

Each keypress prints the new gain dict to stdout. --vocals 0 (etc.) still sets
the starting value from the command line.

Two modes:
  offline (no audio device, safe to test):
      python separate_stream.py --model M.onnx --in-file song.mp3 --out-file out.wav
  live:
      python separate_stream.py --model M.onnx
"""

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- mix knobs
GAINS = {"drums": 1.0, "bass": 1.0, "other": 1.0, "vocals": 1.0}

# Live control. The two key banks sit directly above/below each other on a
# QWERTY board, one column per stem, so muscle memory maps to the mixer:
#     u i o p   -> +5%   drums bass other vocals
#     j k l ;   -> -5%
KEYS_UP = "uiop"
KEYS_DOWN = "jkl;"
STEP = 0.05
GAIN_MIN, GAIN_MAX = 0.0, 2.0
# ----------------------------------------------------------------------------

SOURCES = ["drums", "bass", "other", "vocals"]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Separator:
    """Wraps the torch front/back end around the ONNX core."""

    def __init__(self, model_path, repo, threads=1, lean=True):
        sys.path.insert(0, str(Path(repo) / "vendor"))
        sys.path.insert(0, str(Path(repo) / "profile"))
        import onnxruntime as ort
        import torch as th
        from demucs.htdemucs import HTDemucs
        from onnx_export import front, reconstruct

        self.th, self.front, self.reconstruct = th, front, reconstruct

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if lean:
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
        self.sess = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"]
        )

        # The input shape tells us the window the graph was exported at. Trust
        # the model, never a CLI flag, or the STFT frames will not line up.
        shp = {i.name: i.shape for i in self.sess.get_inputs()}
        self.win = int(shp["mix_t"][2])

        # HTDemucs instance used ONLY for its STFT/mask/iSTFT helpers, which
        # have no learned parameters. Built at the student's width so segment
        # and nfft match the exported graph; the weights are never used.
        from fractions import Fraction

        self.m = HTDemucs(
            sources=SOURCES,
            channels=24,
            bottom_channels=256,
            t_layers=4,
            segment=Fraction(self.win, 44100).limit_denominator(100000),
        ).eval()
        self.sr = self.m.samplerate
        self.ch = self.m.audio_channels
        log(
            f"model {Path(model_path).name}: window {self.win} samples "
            f"({self.win/self.sr:.2f}s) @ {self.sr}Hz"
        )

    def separate(self, chunk):
        """chunk: float32 [ch, win] -> stems float32 [4, ch, win]"""
        th = self.th
        with th.no_grad():
            mix = th.from_numpy(chunk[None])  # [1, ch, win]
            ctx = self.front(self.m, mix)
            x, xt = self.sess.run(
                None, {"mag": ctx["x"].numpy(), "mix_t": ctx["xt"].numpy()}
            )
            out = self.reconstruct(
                self.m, th.from_numpy(x), th.from_numpy(xt), ctx
            )
        return out[0].numpy()  # [4, ch, win]


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

    def __init__(self, sep, gains, stride_s, lookahead_s, xfade_s, preroll_s):
        self.sep = sep
        self.win = sep.win
        sr = sep.sr
        self.stride = int(stride_s * sr)
        self.look = int(lookahead_s * sr)
        self.xfade = int(xfade_s * sr)
        self.preroll_s = preroll_s

        if self.stride + self.look + self.xfade > self.win:
            raise SystemExit("stride + lookahead + xfade must fit inside the "
                             f"{self.win/sr:.2f}s window")

        self.gain_map = dict(gains)
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
        self.outq = queue.Queue(maxsize=64)
        self.done = threading.Event()

        self.held = None                      # xfade tail from the last block
        f = np.linspace(0.0, 1.0, self.xfade, dtype=np.float32)[None, :]
        self.fade_in, self.fade_out = f, 1.0 - f

        self.processed = 0
        self.t_proc = 0.0

    # ----------------------------------------------------------------- mix
    def _rebuild_gains(self):
        # Replacing the whole array is an atomic rebind, so the worker either
        # sees the old gains or the new ones -- never a half-updated mix.
        self.gains = np.array(
            [self.gain_map[s] for s in SOURCES], dtype=np.float32
        )[:, None, None]

    def bump(self, source, delta):
        g = min(GAIN_MAX, max(GAIN_MIN, self.gain_map[source] + delta))
        self.gain_map[source] = round(g, 4)
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
                    break                        # input ended mid-window
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
            stems = self.sep.separate(window)          # [4, ch, win]
            mixed = (stems * self.gains).sum(0)        # [ch, win]
            self.t_proc += time.perf_counter() - t0
            self.processed += 1

            # Tail slice, stopping `look` short of the raw edge. Length is
            # stride+xfade so the extra tail can be blended into the next block.
            end = self.win - self.look
            seg = mixed[:, end - self.stride - self.xfade : end].copy()

            if self.held is not None and self.xfade:
                # same absolute time range as the previous block's held tail
                seg[:, : self.xfade] = (
                    seg[:, : self.xfade] * self.fade_in
                    + self.held * self.fade_out
                )
            out = seg[:, : self.stride] if self.held is not None else seg[
                :, : self.stride
            ]
            self.held = seg[:, self.stride :].copy() if self.xfade else None
            self.outq.put(out.copy())

            if self.processed % 5 == 0:
                rtf = self.t_proc / (self.processed * self.stride / self.sep.sr)
                warn = "  !! CANNOT KEEP UP" if rtf > 1 else ""
                log(f"  blocks={self.processed}  proc/stride={rtf:.2f}x{warn}")
        self.outq.put(None)


def startup_bar(stream, sep, started, initial_rtf=0.5):
    """Progress bar for the dead time before the first audio comes out.

    With tail emission the 7.8s window costs nothing -- it is all lookback --
    so the wait is only: collect one stride, run the model once, then hold
    `preroll` seconds of jitter margin. proc is unknown until a window has
    actually run, so start from an assumed RTF and correct the total as soon
    as the first real measurement lands.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        log("(install tqdm for a progress bar)")
        started.wait()
        return

    win_s = stream.win / sep.sr
    stride_s = stream.stride / sep.sr
    look_s = stream.look / sep.sr

    def estimate(proc):
        return stride_s + stream.preroll_s + proc

    total = estimate(initial_rtf * win_s)
    bar = tqdm(
        total=round(total, 1), unit="s", dynamic_ncols=True,
        bar_format="{desc} {bar} {n:.1f}/{total:.1f}s {postfix}",
        desc="waiting for first block",
    )
    t0 = time.time()
    refined = False
    while not started.wait(0.25):
        if not refined and stream.processed >= 1:
            proc = stream.t_proc / stream.processed
            total = estimate(proc)
            bar.total = round(total, 1)
            bar.set_postfix_str(f"proc {proc:.1f}s/window")
            refined = True
        bar.n = min(round(time.time() - t0, 1), bar.total)
        bar.refresh()
    bar.n = bar.total
    bar.close()

    proc = stream.t_proc / max(stream.processed, 1)
    offset = stride_s + look_s + stream.preroll_s + proc
    log(f"output live after {time.time()-t0:.1f}s -- audio is "
        f"{offset:.1f}s behind the input and stays there "
        f"({stride_s:.1f}s block + {look_s:.1f}s lookahead + "
        f"{stream.preroll_s:.1f}s jitter + {proc:.1f}s inference; "
        f"the {win_s:.1f}s window itself costs nothing)")


def keyboard(stream, stop):
    """Read single keypresses and nudge stem gains live.

    cbreak (not raw) so keys arrive unbuffered without swallowing Ctrl-C, and
    the old termios state is always restored -- otherwise a crash leaves the
    user's shell with no echo.
    """
    import select
    import termios
    import tty

    if not sys.stdin.isatty():
        log("stdin is not a tty -- live gain keys disabled")
        return

    def show(g):
        body = ", ".join(f"'{k}': {v:.2f}" for k, v in g.items())
        print("{" + body + "}", flush=True)

    log(f"keys: {'/'.join(KEYS_UP)} = +{STEP:.0%}   "
        f"{'/'.join(KEYS_DOWN)} = -{STEP:.0%}   "
        f"(drums bass other vocals)   r = reset   q = quit")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            c = sys.stdin.read(1)
            if c in ("q", ""):
                stop.set()
                break
            if c == "r":
                for k in SOURCES:
                    stream.gain_map[k] = 1.0
                stream._rebuild_gains()
                show(stream.gain_map)
            elif c in KEYS_UP:
                show(stream.bump(SOURCES[KEYS_UP.index(c)], STEP))
            elif c in KEYS_DOWN:
                show(stream.bump(SOURCES[KEYS_DOWN.index(c)], -STEP))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def to_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


def read_audio(path, sr, ch):
    """Decode any input to float32 [ch, n] via ffmpeg.

    Deliberately NOT demucs.audio.AudioFile: that module imports lameenc at
    import time (for mp3 *writing*, which we never do), so it drags an extra
    native dependency onto the Pi for no benefit. ffmpeg is already required.
    """
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le",
           "-ar", str(sr), "-ac", str(ch), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    a = np.frombuffer(raw, dtype="<f4").reshape(-1, ch).T
    return np.ascontiguousarray(a)


def run_offline(stream, in_file, out_file, sep):
    import soundfile as sf

    wav = read_audio(in_file, sep.sr, sep.ch)
    log(f"input {Path(in_file).name}: {wav.shape[1]/sep.sr:.1f}s")

    t = threading.Thread(target=stream.worker, daemon=True)
    t.start()

    out = []

    def drain():
        while True:
            blk = stream.outq.get()
            if blk is None:
                break
            out.append(blk)

    d = threading.Thread(target=drain, daemon=True)
    d.start()

    # Feed in stride-sized chunks, exactly as the live reader does, so offline
    # output is bit-for-bit what the stream would produce. Trailing pad lets
    # the last real audio reach the emitted tail region.
    pad = stream.stride + stream.look + stream.xfade
    wav = np.concatenate(
        [wav, np.zeros((sep.ch, pad), dtype=np.float32)], axis=1
    )
    for i in range(0, wav.shape[1], stream.stride):
        stream.feed(wav[:, i : i + stream.stride])
    stream.finish_input()
    t.join()
    d.join()

    y = np.concatenate(out, axis=1) if out else np.zeros((sep.ch, 0))
    sf.write(out_file, y.T, sep.sr, subtype="PCM_16")
    log(f"wrote {out_file}  ({y.shape[1]/sep.sr:.1f}s)")


def run_live(stream, sep, device, block):
    started = threading.Event()   # set by the writer when playback begins
    rec = subprocess.Popen(
        ["arecord", "-D", device, "-f", "S16_LE", "-r", str(sep.sr),
         "-c", str(sep.ch), "-t", "raw", "--period-size", str(block)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    play = subprocess.Popen(
        ["aplay", "-D", device, "-f", "S16_LE", "-r", str(sep.sr),
         "-c", str(sep.ch), "-t", "raw"],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    def reader():
        nbytes = block * sep.ch * 2
        while True:
            data = rec.stdout.read(nbytes)
            if not data:
                break
            a = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            stream.feed(a.reshape(-1, sep.ch).T)
        stream.finish_input()

    def writer():
        # Preroll: hold back `preroll` blocks so one slow window never starves
        # the DAC mid-phrase. This IS the delay buffer.
        # Wait for the first block, then hold an extra `preroll_s` before
        # starting. Delaying by a fraction of a block buys jitter margin at
        # 1:1 in latency, instead of paying a whole stride for it.
        first = stream.outq.get()
        if first is None:
            started.set()
            return
        time.sleep(stream.preroll_s)
        started.set()
        buf = [first]
        for blk in buf:
            play.stdin.write(to_int16(blk.T.ravel()).tobytes())
        while True:
            blk = stream.outq.get()
            if blk is None:
                break
            try:
                play.stdin.write(to_int16(blk.T.ravel()).tobytes())
            except BrokenPipeError:
                break

    stop = threading.Event()
    threads = [
        threading.Thread(target=f, daemon=True)
        for f in (reader, stream.worker, writer,
                  lambda: startup_bar(stream, sep, started),
                  lambda: keyboard(stream, stop))
    ]
    for t in threads:
        t.start()
    log("running -- Ctrl-C to stop")
    try:
        while threads[2].is_alive():
            threads[2].join(0.5)
    except KeyboardInterrupt:
        log("\nstopping ...")
    finally:
        rec.terminate()
        try:
            play.stdin.close()
        except Exception:
            pass
        play.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="simplified student .onnx")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent),
                    help="dir containing vendor/ and profile/")
    ap.add_argument("--threads", type=int, default=4,
                    help="ORT threads. 1 was the optimisation target, but for "
                         "live playback use the cores you have -- headroom "
                         "against underruns beats a benchmark number")
    ap.add_argument("--stride", type=float, default=4.0,
                    help="seconds of audio emitted per model run. MUST exceed "
                         "the per-window inference time (~3.1s on the Pi) or "
                         "the stream falls behind permanently")
    ap.add_argument("--lookahead", type=float, default=1.3,
                    help="stop this far short of the window's right edge. "
                         "Measured: the last 0.65s is ~7dB worse than centre, "
                         "1.3s back recovers most of it")
    ap.add_argument("--xfade", type=float, default=0.1,
                    help="crossfade between consecutive blocks (they come from "
                         "different model runs, so the join needs smoothing)")
    ap.add_argument("--preroll", type=float, default=1.0,
                    help="SECONDS of output buffered before playback starts. "
                         "Jitter margin -- costs exactly its own value in "
                         "latency, unlike buffering a whole block")
    ap.add_argument("--device", default="plughw:CARD=Zero,DEV=0")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--in-file", default="")
    ap.add_argument("--out-file", default="")
    for s in SOURCES:
        ap.add_argument(f"--{s}", type=float, default=GAINS[s],
                        help=f"gain for {s} (0 mutes it)")
    args = ap.parse_args()

    gains = {s: getattr(args, s) for s in SOURCES}
    log(f"stem gains: {gains}")

    sep = Separator(args.model, args.repo, threads=args.threads)
    stream = Stream(sep, gains, args.stride, args.lookahead, args.xfade,
                    args.preroll)
    log(
        f"window {stream.win/sep.sr:.2f}s (all lookback)  "
        f"stride {args.stride:.2f}s  lookahead {args.lookahead:.2f}s  "
        f"xfade {args.xfade:.2f}s  preroll {args.preroll:.2f}s  "
        f"threads {args.threads}"
    )
    log(f"expected latency ~= proc + {args.stride + args.lookahead + args.preroll:.1f}s"
        f"  (proc measured after the first block)")

    if args.in_file:
        if not args.out_file:
            sys.exit("--in-file needs --out-file")
        run_offline(stream, args.in_file, args.out_file, sep)
    else:
        run_live(stream, sep, args.device, args.block)


if __name__ == "__main__":
    main()
