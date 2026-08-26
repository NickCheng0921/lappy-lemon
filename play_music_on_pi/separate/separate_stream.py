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

The per-stem GAINS are the hook for live mixing later -- set vocals to 0 for an
instant karaoke feed, or solo the drums, without touching anything else.

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


def triangular(n):
    """demucs-style window: peaks in the middle, where the model is most
    accurate, so overlap-add trusts window centres over their edges."""
    half = n // 2
    w = np.concatenate(
        [np.arange(1, half + 1), np.arange(n - half, 0, -1)]
    ).astype(np.float32)
    return w / w.max()


class Stream:
    def __init__(self, sep, overlap, gains, preroll):
        self.sep = sep
        self.win = sep.win
        self.stride = max(1, int(self.win * (1.0 - overlap)))
        self.gains = np.array(
            [gains[s] for s in SOURCES], dtype=np.float32
        )[:, None, None]
        self.wnd = triangular(self.win)[None, :]  # [1, win]
        self.preroll = preroll

        self.inbuf = np.zeros((sep.ch, 0), dtype=np.float32)
        self.incond = threading.Condition()
        self.outq = queue.Queue(maxsize=64)
        self.done = threading.Event()

        # overlap-add accumulators
        self.acc = np.zeros((sep.ch, self.win), dtype=np.float32)
        self.accw = np.zeros((1, self.win), dtype=np.float32)

        self.processed = 0
        self.t_proc = 0.0

    # ---------------------------------------------------------------- input
    def feed(self, block):
        with self.incond:
            self.inbuf = np.concatenate([self.inbuf, block], axis=1)
            self.incond.notify()

    def finish_input(self):
        self.done.set()
        with self.incond:
            self.incond.notify_all()

    # --------------------------------------------------------------- worker
    def worker(self):
        while True:
            with self.incond:
                while self.inbuf.shape[1] < self.win and not self.done.is_set():
                    self.incond.wait(timeout=0.5)
                if self.inbuf.shape[1] < self.win:
                    if self.done.is_set():
                        break  # not enough left for a full window
                    continue
                chunk = self.inbuf[:, : self.win].copy()
                self.inbuf = self.inbuf[:, self.stride :]

            t0 = time.perf_counter()
            stems = self.sep.separate(chunk)  # [4, ch, win]
            mixed = (stems * self.gains).sum(0)  # [ch, win]
            self.t_proc += time.perf_counter() - t0
            self.processed += 1

            # overlap-add with the triangular window
            self.acc += mixed * self.wnd
            self.accw += self.wnd

            w = np.maximum(self.accw[:, : self.stride], 1e-6)
            self.outq.put((self.acc[:, : self.stride] / w).copy())

            # slide accumulators forward by one stride
            pad = np.zeros((self.acc.shape[0], self.stride), dtype=np.float32)
            self.acc = np.concatenate([self.acc[:, self.stride :], pad], axis=1)
            self.accw = np.concatenate(
                [
                    self.accw[:, self.stride :],
                    np.zeros((1, self.stride), dtype=np.float32),
                ],
                axis=1,
            )

            if self.processed % 5 == 0:
                rtf = self.t_proc / (self.processed * self.win / self.sep.sr)
                warn = "  !! SLOWER THAN REAL TIME" if rtf > 1 else ""
                log(f"  windows={self.processed}  avg RTF={rtf:.2f}x{warn}")
        self.outq.put(None)


def startup_bar(stream, sep, started, initial_rtf=0.5):
    """Progress bar for the dead time before the first audio comes out.

    Time-to-first-sound is predictable, so show it rather than leaving the user
    guessing. Block k can only exist once its window has been captured (capture
    is real-time) and then processed:

        t(block k) = window + k*stride + proc
        playback starts at block (preroll-1)

    proc is unknown until a window has actually run, so start from an assumed
    RTF and correct the total as soon as the first real measurement lands.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        log("(install tqdm for a progress bar)")
        started.wait()
        return

    win_s = stream.win / sep.sr
    stride_s = stream.stride / sep.sr

    def estimate(proc):
        return win_s + (stream.preroll - 1) * stride_s + proc

    total = estimate(initial_rtf * win_s)
    bar = tqdm(total=round(total, 1), unit="s", dynamic_ncols=True,
               bar_format="{desc} {bar} {n:.1f}/{total:.1f}s {postfix}",
               desc="filling delay buffer")
    t0 = time.time()
    refined = False
    while not started.wait(0.25):
        if not refined and stream.processed >= 1:
            proc = stream.t_proc / stream.processed
            total = estimate(proc)
            bar.total = round(total, 1)
            bar.set_postfix_str(f"proc {proc:.1f}s/window, RTF {proc/win_s:.2f}x")
            refined = True
        bar.n = min(round(time.time() - t0, 1), bar.total)
        bar.refresh()
    bar.n = bar.total
    bar.close()
    log(f"output live after {time.time()-t0:.1f}s "
        f"(audio you hear is ~{win_s + (stream.preroll-1)*stride_s:.0f}s behind "
        f"the input, and stays that far behind)")


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
    # pad so the tail still forms a full final window
    wav = np.concatenate(
        [wav, np.zeros((sep.ch, stream.win), dtype=np.float32)], axis=1
    )
    stream.feed(wav)
    stream.finish_input()

    out = []
    while True:
        blk = stream.outq.get()
        if blk is None:
            break
        out.append(blk)
    t.join()

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
        buf = []
        while len(buf) < stream.preroll:
            blk = stream.outq.get()
            if blk is None:
                break
            buf.append(blk)
        started.set()
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

    threads = [
        threading.Thread(target=f, daemon=True)
        for f in (reader, stream.worker, writer,
                  lambda: startup_bar(stream, sep, started))
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
    ap.add_argument("--overlap", type=float, default=0.1,
                    help="window overlap. 0 is fastest but seams click; 0.1 "
                         "measured 0.88x e2e, still faster than real time")
    ap.add_argument("--preroll", type=int, default=2,
                    help="output blocks buffered before playback starts")
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
    stream = Stream(sep, args.overlap, gains, args.preroll)
    log(
        f"window {stream.win/sep.sr:.2f}s  stride {stream.stride/sep.sr:.2f}s "
        f"(overlap {args.overlap})  threads {args.threads}"
    )

    if args.in_file:
        if not args.out_file:
            sys.exit("--in-file needs --out-file")
        run_offline(stream, args.in_file, args.out_file, sep)
    else:
        run_live(stream, sep, args.device, args.block)


if __name__ == "__main__":
    main()
