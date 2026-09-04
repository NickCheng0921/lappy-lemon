"""Five slide pots on two ADS1115 ADCs, read over i2c."""

import os
import time

from gains import (
    GAIN_MAX_DB,
    MUTE_FLOOR_DB,
    POT_MUTE_POS,
    POT_STEPS,
    POT_UNITY_POS,
    SOURCES,
    detent,
)
from util import log

try:
    from fader_viz import FaderViz
except ImportError:  # display is cosmetic; the app is not
    FaderViz = None

POT_MAP_DEFAULT = "0x48:0,1,2,3 0x49:0"

POT_SMOOTH = 0.5  # one-pole EMA on position; 1.0 disables smoothing

I2C_SLAVE = 0x0703
ADS_REG_CONV, ADS_REG_CONFIG = 0x00, 0x01
ADS_MUX_SINGLE = {0: 0x4, 1: 0x5, 2: 0x6, 3: 0x7}
ADS_PGA = 0x1  # +/-4.096V full scale, clears a 3.3V rail with room
ADS_FSR_VOLTS = 4.096
ADS_DR_CODE, ADS_DR_SPS = 4, 128

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
