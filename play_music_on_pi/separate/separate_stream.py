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

Stem gains are adjustable live from the keyboard while it runs. Steps are in
decibels, so one press sounds like the same size move wherever the fader sits:

    7 8 9 0   mute toggle    drums  bass  other  vocals
    u i o p   +1.5 dB
    j k l ;   -1.5 dB
    r         reset every stem to unity
    q         quit

Each keypress prints the new gains to stdout as both a linear multiplier and
dB. --vocals 0 (etc.) still sets the starting value from the command line, and
stays a linear multiplier there: 1.0 = unity, 0 = silence.

Two modes:
  offline (no audio device, safe to test):
      python separate_stream.py --model M.onnx --in-file song.mp3 --out-file out.wav
  live:
      python separate_stream.py --model M.onnx
"""

import argparse
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- mix knobs
GAINS = {"drums": 1.0, "bass": 1.0, "other": 1.0, "vocals": 1.0}

# Live control. The key banks sit directly above/below each other on a QWERTY
# board, one column per stem, so muscle memory maps to the mixer:
#     7 8 9 0   -> mute toggle   drums bass other vocals
#     u i o p   -> louder
#     j k l ;   -> quieter
KEYS_MUTE = "7890"
KEYS_UP = "uiop"
KEYS_DOWN = "jkl;"

# Steps are multiplicative (in dB), not additive. Loudness tracks dB, so a
# fixed *amplitude* step is wildly uneven: +0.1 is 6 dB near silence but only
# 0.4 dB near the top of the range, i.e. one press did 14x more at one end of
# the fader than the other. 1.5 dB is a bit above the just-noticeable
# difference for a complex signal, so every press is audible and none is a
# jump.
STEP_DB = 1.5
GAIN_MAX_DB = 12.0     # boost ceiling; the 4 stems are summed and hard-clipped
                       # at +/-1.0 in to_int16(), so more than this is mush

# A dB fader never reaches zero on its own, so silence has to be a decision
# rather than a limit. Hold the press count instead of the depth: ten presses
# down from unity kills a stem, the ninth is the quietest audible setting, and
# a press back up from silence returns there. Deriving the floor from the step
# keeps that count fixed if STEP_DB ever changes.
PRESSES_TO_MUTE = 10
MUTE_FLOOR_DB = -STEP_DB * (PRESSES_TO_MUTE - 1)
GAIN_MIN, GAIN_MAX = 0.0, 10.0 ** (GAIN_MAX_DB / 20.0)
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
        self.premute = {}                     # gain to restore on un-mute
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
        self.outq = queue.Queue(maxsize=8)   # blocks hold 4 stems each
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

    def mixdown(self, stems_block):
        """[4, ch, n] -> [ch, n] using the CURRENT gains. Called per output
        chunk by the writer, which is what makes the keys feel instant."""
        return (stems_block * self.gains).sum(0)

    def bump(self, source, steps):
        """Move a stem by `steps` * STEP_DB decibels.

        Multiplicative, so the move is the same perceived size at any fader
        position. Silence is a special case at both ends: log10(0) has no
        answer, so a stem stepped below MUTE_FLOOR_DB snaps to zero instead of
        decaying forever, and a stem at zero re-enters the scale at that same
        floor -- see PRESSES_TO_MUTE.
        """
        g = self.gain_map[source]
        # Silence sits one step *below* the floor, so a down-press and the
        # up-press undoing it are exact inverses: -13.5 dB -> mute -> -13.5 dB.
        # Parking it at the floor instead would skip the quietest setting on
        # the way back up.
        db = (MUTE_FLOOR_DB - STEP_DB) if g <= 0.0 else 20.0 * math.log10(g)
        # Quantise in dB, the unit the limits are written in. The fader's
        # position lives in gain_map as a multiplier, so every press round
        # trips through log10/10**; rounding the *gain* instead let that error
        # land a press a hair below MUTE_FLOOR_DB and mute a step early, now
        # that the floor sits exactly on a step boundary.
        db = round(db + steps * STEP_DB, 3)
        g = GAIN_MIN if db < MUTE_FLOOR_DB else min(GAIN_MAX, 10.0 ** (db / 20.0))
        self.gain_map[source] = g
        self._rebuild_gains()
        return self.gain_map

    def toggle_mute(self, source):
        """Silence a stem, or restore whatever it was before the mute.

        Stepping down still takes PRESSES_TO_MUTE presses, which is too slow
        to kill a stem mid-song, and it forgets where the fader was.
        """
        if self.gain_map[source] > 0.0:
            self.premute[source] = self.gain_map[source]
            self.gain_map[source] = GAIN_MIN
        else:
            self.gain_map[source] = self.premute.get(source, 1.0)
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
                    seg[:, :, : self.xfade] * self.fade_in
                    + self.held * self.fade_out
                )
            self.held = seg[:, :, self.stride :].copy() if self.xfade else None
            self.outq.put(seg[:, :, : self.stride].copy())

            if self.processed % 5 == 0:
                rtf = self.t_proc / (self.processed * self.stride / self.sep.sr)
                warn = "  !! CANNOT KEEP UP" if rtf > 1 else ""
                log(f"  blocks={self.processed}  proc/stride={rtf:.2f}x{warn}")
        self.outq.put(None)


def startup_bar(stream, sep, started, initial_rtf=0.5):
    """Plain-text progress line for the wait before the first audio.

    Hand-rolled rather than tqdm: the Pi ships tqdm 4.70, which raises
    TypeError formatting a custom bar_format when set_postfix_str triggers a
    refresh. A cosmetic progress bar is not worth a dependency that can kill a
    thread, so this writes the line itself.

    With tail emission the 7.8s window costs nothing (all lookback), so the
    wait is: collect one stride, run the model once, hold `preroll` of jitter.
    """
    win_s = stream.win / sep.sr
    stride_s = stream.stride / sep.sr
    look_s = stream.look / sep.sr

    def estimate(proc):
        return stride_s + stream.preroll_s + proc

    total = estimate(initial_rtf * win_s)     # refined once proc is known
    width = 34
    t0 = time.time()
    while not started.wait(0.25):
        if stream.processed >= 1:
            total = estimate(stream.t_proc / stream.processed)
        el = time.time() - t0
        frac = min(1.0, el / max(total, 1e-6))
        fill = int(width * frac)
        sys.stderr.write(
            f"\r  waiting for first block [{'#'*fill}{'.'*(width-fill)}] "
            f"{el:4.1f}/{total:4.1f}s"
        )
        sys.stderr.flush()
    sys.stderr.write("\r" + " " * (width + 44) + "\r")
    sys.stderr.flush()

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
        # Both units: dB is what the ear reads, the multiplier is what the
        # mixdown actually does and what --vocals etc. take.
        body = "  ".join(
            f"{k} {'mute' if v <= 0.0 else f'{20 * math.log10(v):+5.1f}dB'}"
            f" ({v:.2f})"
            for k, v in g.items()
        )
        print(body, flush=True)

    log(f"keys: {'/'.join(KEYS_UP)} = +{STEP_DB}dB   "
        f"{'/'.join(KEYS_DOWN)} = -{STEP_DB}dB   "
        f"{'/'.join(KEYS_MUTE)} = mute   "
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
                stream.premute.clear()
                stream._rebuild_gains()
                show(stream.gain_map)
            elif c in KEYS_UP:
                show(stream.bump(SOURCES[KEYS_UP.index(c)], +1))
            elif c in KEYS_DOWN:
                show(stream.bump(SOURCES[KEYS_DOWN.index(c)], -1))
            elif c in KEYS_MUTE:
                show(stream.toggle_mute(SOURCES[KEYS_MUTE.index(c)]))
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

    # queue carries stems; apply the (static, CLI-set) gains here
    out = [stream.mixdown(b) for b in out]
    y = np.concatenate(out, axis=1) if out else np.zeros((sep.ch, 0))
    sf.write(out_file, y.T, sep.sr, subtype="PCM_16")
    log(f"wrote {out_file}  ({y.shape[1]/sep.sr:.1f}s)")


def run_live(stream, sep, device, block, chunk):
    started = threading.Event()   # set by the writer when playback begins
    rec = subprocess.Popen(
        ["arecord", "-D", device, "-f", "S16_LE", "-r", str(sep.sr),
         "-c", str(sep.ch), "-t", "raw", "--period-size", str(block)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    # --buffer-time caps ALSA's own queue. Without it aplay can sit on a
    # second of audio, which would delay gain changes no matter how finely
    # we mix them.
    play = subprocess.Popen(
        ["aplay", "-D", device, "-f", "S16_LE", "-r", str(sep.sr),
         "-c", str(sep.ch), "-t", "raw",
         "--buffer-time", "200000", "--period-time", "50000"],
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
        def emit(blk):
            # Slice into small pieces and re-read the gains for each one.
            # Writing a whole 4s block at once would quantise gain changes to
            # the block boundary; at `chunk` samples the response is ~50ms.
            n = blk.shape[2]
            for i in range(0, n, chunk):
                mixed = stream.mixdown(blk[:, :, i : i + chunk])
                play.stdin.write(to_int16(mixed.T.ravel()).tobytes())

        try:
            for blk in buf:
                emit(blk)
            while True:
                blk = stream.outq.get()
                if blk is None:
                    break
                emit(blk)
        except BrokenPipeError:
            pass

    stop = threading.Event()
    threads = [
        threading.Thread(target=f, daemon=True)
        for f in (reader, stream.worker, writer,
                  lambda: startup_bar(stream, sep, started),
                  lambda: keyboard(stream, stop))
    ]
    for t in threads:
        t.start()
    log("running -- Ctrl-C or q to stop")
    try:
        while threads[2].is_alive() and not stop.is_set():
            threads[2].join(0.5)
    except KeyboardInterrupt:
        log("stopping ...")
    finally:
        # Ordered shutdown. Killing the worker while ORT is mid-Run aborts
        # in C++ ("terminate called without an active exception", preceded
        # by a bogus Softmax/GetElementType error). So: stop capture, let
        # the worker finish the window it is on, then tear down playback.
        stop.set()
        stream.finish_input()
        try:
            rec.terminate()
        except Exception:
            pass
        worker_t = threads[1]
        worker_t.join(timeout=(stream.win / sep.sr) + 3.0)
        if worker_t.is_alive():
            log("worker still busy at exit -- ORT may warn on teardown")
        try:
            play.stdin.close()
        except Exception:
            pass
        try:
            play.wait(timeout=3)
        except Exception:
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
    ap.add_argument("--block", type=int, default=4096,
                    help="capture read size (samples)")
    ap.add_argument("--out-chunk", type=int, default=2048,
                    help="playback mix granularity (samples). Gains are "
                         "re-read per chunk, so this sets how fast the keys "
                         "respond: 2048 @ 44.1kHz = 46ms")
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
        run_live(stream, sep, args.device, args.block, args.out_chunk)


if __name__ == "__main__":
    main()
