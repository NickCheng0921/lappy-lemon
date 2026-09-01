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

# ----------------------------------------------------------------- mix knobs
GAINS = {"drums": 1.0, "bass": 1.0, "other": 1.0, "vocals": 1.0}

# The travel is mapped in dB, not in amplitude. Loudness tracks dB, so a fader
# that is linear in amplitude is wildly uneven: the same physical distance is
# 6 dB near silence and 0.4 dB near the ceiling, i.e. 14x more effect at one
# end of the slider than the other. STEP_DB survives the move from keys to
# pots because MUTE_FLOOR_DB is still derived from it.
STEP_DB = 1.5
GAIN_MAX_DB = 12.0  # boost ceiling; the 4 stems are summed and hard-clipped
# at +/-1.0 in to_int16(), so more than this is mush

# A dB fader never reaches zero on its own, so silence has to be a decision
# rather than a limit. The keyboard build counted presses (ten down from unity
# kills a stem, the ninth is the quietest audible setting); a pot has no press
# count, so POT_MUTE_POS declares the bottom of the travel off and the usable
# scale ends one notch above it, here, at MUTE_FLOOR_DB.
PRESSES_TO_MUTE = 10
MUTE_FLOOR_DB = -STEP_DB * (PRESSES_TO_MUTE - 1)
GAIN_MIN, GAIN_MAX = 0.0, 10.0 ** (GAIN_MAX_DB / 20.0)
# ----------------------------------------------------------------------------

SOURCES = ["drums", "bass", "other", "vocals"]

# ------------------------------------------------------------------- faders
# Five slide pots on two ADS1115s (4 channels each, addresses set by the ADDR
# pin: GND=0x48, VDD=0x49). Unused ADC channels must be tied to GND -- a
# floating input reads as a convincing mid-scale position, not as zero.
POT_MAP_DEFAULT = "0x48:0,1,2,3 0x49:0"
POT_UNITY_POS = 0.80  # travel fraction that means unity gain
POT_MUTE_POS = 0.02  # below this a fader is off, not merely quiet
POT_SMOOTH = 0.5  # one-pole EMA on position; 1.0 disables smoothing

# Travel is quantised into detents, so ADC noise cannot move the gain at all.
# Hysteresis is what makes that work -- without it a fader parked on a boundary
# flips between two detents, which is worse than the jitter it replaces.
POT_STEPS = 20  # 5% of travel each
POT_HYST = 0.25  # must overshoot the midpoint by this much to move

I2C_SLAVE = 0x0703
ADS_REG_CONV, ADS_REG_CONFIG = 0x00, 0x01
ADS_MUX_SINGLE = {0: 0x4, 1: 0x5, 2: 0x6, 3: 0x7}
ADS_PGA = 0x1  # +/-4.096V full scale, clears a 3.3V rail with room
ADS_FSR_VOLTS = 4.096
ADS_DR_CODE, ADS_DR_SPS = 4, 128


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
            out = self.reconstruct(self.m, th.from_numpy(x), th.from_numpy(xt), ctx)
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


def detent(pos, held, steps=POT_STEPS, hyst=POT_HYST):
    """Snap position 0..1 to one of `steps` detents. `held` is the latched
    index (None first time); noise below 0.5+hyst steps cannot dislodge it."""
    x = min(max(pos, 0.0), 1.0) * steps
    if held is None:
        return int(round(x))
    if abs(x - held) > 0.5 + hyst:
        return int(min(max(round(x), 0), steps))
    return held


def pot_gain(pos):
    """Fader position (0..1) -> linear gain, on the range the keys used.

    The top of the travel above POT_UNITY_POS is boost (0 .. GAIN_MAX_DB) and
    everything below it is cut (MUTE_FLOOR_DB .. 0). Both halves are linear in
    dB rather than in amplitude, for the reason STEP_DB exists: a fixed
    amplitude move is 6 dB near silence and 0.4 dB near the ceiling, so a
    linear-in-amplitude fader would do 14x more at one end than the other.

    Silence stays a decision rather than a limit, as it was for the keys.
    A dB fader never reaches zero, so the bottom POT_MUTE_POS of the travel is
    declared off; the usable scale ends one notch above it at MUTE_FLOOR_DB.
    """
    if pos <= POT_MUTE_POS:
        return GAIN_MIN
    if pos >= POT_UNITY_POS:
        frac = (pos - POT_UNITY_POS) / max(1.0 - POT_UNITY_POS, 1e-9)
        db = min(1.0, frac) * GAIN_MAX_DB
    else:
        frac = (pos - POT_MUTE_POS) / (POT_UNITY_POS - POT_MUTE_POS)
        db = MUTE_FLOOR_DB * (1.0 - frac)
    return min(GAIN_MAX, 10.0 ** (db / 20.0))


class Ads1115:
    """Minimal single-shot ADS1115 reader over raw /dev/i2c-N.

    No smbus2 on the Pi venv and no reason to add one: this is three registers.
    Single-shot rather than continuous because one ADC sits behind a 4-way mux,
    so continuous mode only ever tracks the latched channel.
    """

    def __init__(self, bus, addr):
        import fcntl

        self.addr = addr
        self.fd = os.open("/dev/i2c-" + str(bus), os.O_RDWR)
        # I2C_SLAVE only latches the target address -- it never touches the
        # bus, so an absent board opens fine and fails later with EREMOTEIO.
        # Probe here, where the error can still say something useful.
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)
        try:
            os.write(self.fd, bytes([ADS_REG_CONFIG]))
            os.read(self.fd, 2)
        except OSError:
            os.close(self.fd)
            raise OSError("no ADS1115 answering at 0x%02x on i2c-%d" % (addr, bus))

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass

    def read_channel(self, channel):
        msb = 0x80 | (ADS_MUX_SINGLE[channel] << 4) | (ADS_PGA << 1) | 0x01
        lsb = (ADS_DR_CODE << 5) | 0x03
        os.write(self.fd, bytes([ADS_REG_CONFIG, msb, lsb]))
        time.sleep(1.0 / ADS_DR_SPS)
        deadline = time.time() + 0.25
        while True:
            os.write(self.fd, bytes([ADS_REG_CONFIG]))
            if os.read(self.fd, 2)[0] & 0x80:  # OS bit set = conversion done
                break
            if time.time() > deadline:
                raise TimeoutError("0x%02x ch%d never finished" % (self.addr, channel))
            time.sleep(0.0005)
        os.write(self.fd, bytes([ADS_REG_CONV]))
        hi, lo = os.read(self.fd, 2)
        val = (hi << 8) | lo
        if val & 0x8000:  # 16-bit two's complement
            val -= 0x10000
        return val


def parse_pot_map(spec):
    """'0x48:0,1,2,3 0x49:0' -> [(addr, channel), ...] in fader order."""
    out = []
    for group in spec.split():
        addr_s, _, chans = group.partition(":")
        for c in chans.split(","):
            out.append((int(addr_s, 0), int(c)))
    return out


def pots(stream, stop, cfg, started=None):
    """Poll the five faders and drive the mix. Runs in its own thread.

    A wiring fault must not take the audio down with it: a board that will not
    answer, or a read that fails mid-song, logs and leaves the gains where they
    are instead of raising. Losing fader control is an annoyance; losing the
    stream mid-song is the thing this app exists to avoid.
    """
    if cfg is None:
        return
    try:
        chans = parse_pot_map(cfg["map"])
    except ValueError as e:
        log("bad --pot-map %r (%s) -- faders disabled" % (cfg["map"], e))
        return
    want = len(SOURCES) + 1
    if len(chans) != want:
        log(
            "--pot-map lists %d faders, need %d (%s + master) -- faders off"
            % (len(chans), want, " ".join(SOURCES))
        )
        return

    boards = {}
    try:
        for addr, _ in chans:
            if addr not in boards:
                boards[addr] = Ads1115(cfg["bus"], addr)
    except OSError as e:
        for b in boards.values():
            b.close()
        log("%s -- faders disabled, keeping the command-line gains" % e)
        log("check ADDR (GND=0x48, VDD=0x49) and `sudo i2cdetect -y 1`")
        return

    names = SOURCES + ["master"]
    log(
        "faders: "
        + "  ".join("%s=0x%02x:A%d" % (n, a, c) for n, (a, c) in zip(names, chans))
    )
    log(
        "unity at %.0f%% travel, %+.0fdB at the top, %.1fdB just above %.0f%%, "
        "silent below that"
        % (100 * POT_UNITY_POS, GAIN_MAX_DB, MUTE_FLOOR_DB, 100 * POT_MUTE_POS)
    )

    viz = None
    if FaderViz is not None and cfg.get("viz", "auto") != "off":
        try:
            viz = FaderViz(names, mode=cfg.get("viz", "auto"), unity_pos=POT_UNITY_POS)
        except Exception as e:  # never fatal
            log("fader display off (%s)" % e)

    smooth = [None] * len(chans)
    latch = [None] * len(chans)  # detent index per fader
    shown = False
    sent = [None] * len(chans)
    period = 1.0 / max(cfg["hz"], 1e-3)
    try:
        while not stop.is_set():
            t0 = time.time()
            try:
                for i, (addr, chan) in enumerate(chans):
                    raw = boards[addr].read_channel(chan)
                    volts = raw * ADS_FSR_VOLTS / 32768.0
                    pos = min(max(volts / cfg["vref"], 0.0), 1.0)
                    smooth[i] = (
                        pos
                        if smooth[i] is None
                        else (smooth[i] + POT_SMOOTH * (pos - smooth[i]))
                    )
                    latch[i] = detent(smooth[i], latch[i], cfg["steps"])
            except (OSError, TimeoutError) as e:
                log("fader read failed (%s) -- holding the last gains" % e)
                if viz is not None:
                    viz.invalidate()
                time.sleep(0.5)
                continue
            # Rebuild only when a detent actually changes -- the whole point
            # of quantising is that a resting fader produces no updates.
            quant = [d / cfg["steps"] for d in latch]
            moved = any(
                sent[i] is None or quant[i] != sent[i] for i in range(len(chans))
            )
            if moved:
                sent = quant
                stream.set_positions(sent)
            # The display waits for playback to start: until then startup_bar()
            # owns the console, and two threads redrawing it just fight.
            if viz is not None and (started is None or started.is_set()):
                if moved or not shown:
                    viz.draw(
                        sent, [stream.gain_map[s] for s in SOURCES] + [stream.master]
                    )
                    shown = True
            time.sleep(max(0.0, period - (time.time() - t0)))
    finally:
        for b in boards.values():
            b.close()


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
