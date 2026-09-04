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

Stem gains come from five slide pots on two ADS1115 ADCs, read over i2c-1
(pi_hw_test/five_pot_test.py has the wiring and a bench test):

    fader 0..3   drums  bass  other  vocals
    fader 4      master

A fader is ABSOLUTE where the keys were relative: its position IS the gain, so
there is no mute toggle and no un-mute memory. The travel is mapped in decibels,
the unit the ear reads, over the same range the keys covered:

    top 20%      unity .. +12 dB    boost
    at 80%       unity
    bottom 80%   -13.5 dB .. unity  cut
    bottom 2%    silent

--vocals 0 (etc.) still sets a starting value from the command line, but the
first fader poll overwrites it -- that is what absolute means. The
keyboard-controlled version of this app is separate_keyboard.py.

Two modes:
  offline (no audio device, safe to test):
      python separate_stream.py --model M.onnx --in-file song.mp3 --out-file out.wav
  live:
      python separate_stream.py --model M.onnx
"""

import argparse
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    from fader_viz import FaderViz
except ImportError:  # display is cosmetic; app is not
    FaderViz = None

from audio_io import _cap_pipe, read_audio, to_int16
from faders import POT_MAP_DEFAULT, pots
from gains import GAINS, GAIN_MAX_DB, MUTE_FLOOR_DB, POT_MUTE_POS, POT_STEPS, POT_UNITY_POS, SOURCES
from separator import Separator
from stream import Stream
from util import log


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

    total = estimate(initial_rtf * win_s)  # refined once proc is known
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
    log(
        f"output live after {time.time()-t0:.1f}s -- audio is "
        f"{offset:.1f}s behind the input and stays there "
        f"({stride_s:.1f}s block + {look_s:.1f}s lookahead + "
        f"{stream.preroll_s:.1f}s jitter + {proc:.1f}s inference; "
        f"the {win_s:.1f}s window itself costs nothing)"
    )

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
    wav = np.concatenate([wav, np.zeros((sep.ch, pad), dtype=np.float32)], axis=1)
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

def run_live(stream, sep, device, block, chunk, pipe_bytes, pot_cfg=None):
    started = threading.Event()  # set by the writer when playback begins
    rec = subprocess.Popen(
        [
            "arecord",
            "-D",
            device,
            "-f",
            "S16_LE",
            "-r",
            str(sep.sr),
            "-c",
            str(sep.ch),
            "-t",
            "raw",
            "--period-size",
            str(block),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    # --buffer-time caps ALSA's own queue. Without it aplay can sit on a
    # second of audio, which would delay gain changes no matter how finely
    # we mix them.
    play = subprocess.Popen(
        [
            "aplay",
            "-D",
            device,
            "-f",
            "S16_LE",
            "-r",
            str(sep.sr),
            "-c",
            str(sep.ch),
            "-t",
            "raw",
            "--buffer-time",
            "200000",
            "--period-time",
            "50000",
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    # Shrink the OS pipe to aplay. The default on this Pi is 262144 bytes =
    # 1486ms of stereo s16 -- audio already mixed and committed downstream,
    # which is what a gain change has to wait behind. 16KB leaves ~93ms.
    # Combined with the 200ms ALSA ring that puts key response near 300ms
    # instead of ~1.7s.
    try:
        import fcntl

        F_SETPIPE_SZ, F_GETPIPE_SZ = 1031, 1032
        fcntl.fcntl(play.stdin.fileno(), F_SETPIPE_SZ, 16384)
        got = fcntl.fcntl(play.stdin.fileno(), F_GETPIPE_SZ)
        log(
            f"output pipe {got} bytes "
            f"({1000*got/(sep.sr*sep.ch*2):.0f}ms committed)"
        )
    except Exception as e:
        log(f"could not shrink output pipe ({e}); key response will lag")
    # ...and the pipe INTO aplay needs the same treatment, for the same
    # reason. Everything past mixdown() is committed at the old gain, and
    # Raspberry Pi OS gives a new pipe 256 KiB = 1.49s of stereo S16 -- more
    # than the whole preroll cushion, so without this the cushion drains out
    # of outq (where it is still 4 separate stems a keypress can reach) and
    # sits here as finished PCM instead. Capping the pipe keeps the cushion
    # upstream; ALSA's 200ms buffer is what guards against underruns.
    # Must happen before the writer starts -- shrinking a pipe with data
    # already queued is not reliably allowed.
    pipe_ms = _cap_pipe(play.stdin, pipe_bytes, sep)
    if pipe_ms is not None:
        chunk_ms = chunk / sep.sr * 1000.0
        log(
            f"fader response ~= {chunk_ms + pipe_ms + 200:.0f}ms "
            f"({chunk_ms:.0f} mix chunk + {pipe_ms:.0f} pipe + 200 ALSA); "
            "this is separate from the ~9.5s pipeline latency"
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
        for f in (
            reader,
            stream.worker,
            writer,
            lambda: startup_bar(stream, sep, started),
            lambda: pots(stream, stop, pot_cfg, started),
        )
    ]
    for t in threads:
        t.start()
    log("running -- Ctrl-C to stop")
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
    ap.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parent),
        help="dir containing vendor/ and profile/",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ORT threads. 1 was the optimisation target, but for "
        "live playback use the cores you have -- headroom "
        "against underruns beats a benchmark number",
    )
    ap.add_argument(
        "--stride",
        type=float,
        default=4.0,
        help="seconds of audio emitted per model run. MUST exceed "
        "the per-window inference time (~3.1s on the Pi) or "
        "the stream falls behind permanently",
    )
    ap.add_argument(
        "--lookahead",
        type=float,
        default=1.3,
        help="stop this far short of the window's right edge. "
        "Measured: the last 0.65s is ~7dB worse than centre, "
        "1.3s back recovers most of it",
    )
    ap.add_argument(
        "--xfade",
        type=float,
        default=0.1,
        help="crossfade between consecutive blocks (they come from "
        "different model runs, so the join needs smoothing)",
    )
    ap.add_argument(
        "--max-lag",
        type=float,
        default=2.0,
        help="seconds of backlog before stale audio is dropped to stay "
        "current. Live only -- offline never drops",
    )
    ap.add_argument(
        "--preroll",
        type=float,
        default=1.0,
        help="SECONDS of output buffered before playback starts. "
        "Jitter margin -- costs exactly its own value in "
        "latency, unlike buffering a whole block",
    )
    ap.add_argument("--device", default="plughw:CARD=Zero,DEV=0")
    ap.add_argument(
        "--block", type=int, default=4096, help="capture read size (samples)"
    )
    ap.add_argument(
        "--out-chunk",
        type=int,
        default=2048,
        help="playback mix granularity (samples). Gains are "
        "re-read per chunk, so this sets how fast the keys "
        "respond: 2048 @ 44.1kHz = 46ms",
    )
    ap.add_argument(
        "--pipe-bytes",
        type=int,
        default=16384,
        help="cap on the pipe into aplay. This is finished PCM, "
        "so it delays key response 1:1: 16384 @ 44.1kHz "
        "stereo = 93ms. Raise it if you get underruns",
    )
    ap.add_argument(
        "--pot-map",
        default=POT_MAP_DEFAULT,
        help="fader -> ADC channel map, in the order "
        + " ".join(SOURCES)
        + " master. Addresses come from "
        "the ADS1115 ADDR pin: GND=0x48, VDD=0x49",
    )
    ap.add_argument(
        "--pot-bus",
        type=int,
        default=1,
        help="i2c bus. The DA7212 codec shares it at 0x1a, which "
        "is fine -- i2c is a bus and 0x48/0x49 do not collide",
    )
    ap.add_argument(
        "--pot-vref",
        type=float,
        default=3.295,
        help="volts at the top of fader travel, i.e. what counts "
        "as unity*max. Measured 3.295 on this rig, not 3.300; "
        "read it off five_pot_test.py and set it here",
    )
    ap.add_argument(
        "--pot-steps",
        type=int,
        default=POT_STEPS,
        help="detents across the travel (20 = 5%% each)",
    )
    ap.add_argument(
        "--pot-hz",
        type=float,
        default=20.0,
        help="fader poll rate. A 5-fader scan takes ~40ms at "
        "128 SPS, so much above 20 just spins",
    )
    ap.add_argument(
        "--fader-viz",
        default="auto",
        choices=("auto", "ansi", "plain", "off"),
        help="live fader bars. auto picks ansi only on a tty, and "
        "`ssh host cmd` is not one -- pass ansi explicitly if "
        "a real terminal is reading the other end of the pipe",
    )
    ap.add_argument(
        "--no-pots",
        action="store_true",
        help="ignore the faders and keep the command-line gains",
    )
    ap.add_argument("--in-file", default="")
    ap.add_argument("--out-file", default="")
    for s in SOURCES:
        ap.add_argument(
            f"--{s}", type=float, default=GAINS[s], help=f"gain for {s} (0 mutes it)"
        )
    args = ap.parse_args()

    gains = {s: getattr(args, s) for s in SOURCES}
    log(f"stem gains: {gains}")

    sep = Separator(args.model, args.repo, threads=args.threads)
    stream = Stream(
        sep,
        gains,
        args.stride,
        args.lookahead,
        args.xfade,
        args.preroll,
        max_lag_s=None if args.in_file else args.max_lag,
    )
    log(
        f"window {stream.win/sep.sr:.2f}s (all lookback)  "
        f"stride {args.stride:.2f}s  lookahead {args.lookahead:.2f}s  "
        f"xfade {args.xfade:.2f}s  preroll {args.preroll:.2f}s  "
        f"threads {args.threads}"
    )
    log(
        f"expected latency ~= proc + {args.stride + args.lookahead + args.preroll:.1f}s"
        f"  (proc measured after the first block)"
    )

    if args.in_file:
        if not args.out_file:
            sys.exit("--in-file needs --out-file")
        run_offline(stream, args.in_file, args.out_file, sep)
    else:
        pot_cfg = (
            None
            if args.no_pots
            else {
                "map": args.pot_map,
                "bus": args.pot_bus,
                "vref": args.pot_vref,
                "hz": args.pot_hz,
                "steps": args.pot_steps,
                "viz": args.fader_viz,
            }
        )
        run_live(
            stream,
            sep,
            args.device,
            args.block,
            args.out_chunk,
            args.pipe_bytes,
            pot_cfg,
        )


if __name__ == "__main__":
    main()
