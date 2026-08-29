#!/usr/bin/env python3
"""Read a slide pot through an ADS1115 on i2c-1. No deps -- raw /dev/i2c-1.

Wiring (Pi 5, 40-pin passed up over the DA7212 HAT -- see play_music_on_pi/MISC.md):

    ADS1115         Pi physical pin
    VDD             1   (3.3V)      <- NOT 5V: board says "Max AIN: VDD", and the
    GND             9   (GND)          module's pull-ups would put 5V on SDA/SCL
    SCL             5   (GPIO3/SCL1)
    SDA             3   (GPIO2/SDA1)
    ADDR            9   (GND)       -> address 0x48
    ALRT            open
    A0              <- pot wiper (OTA)

    slide pot (dual gang -- only one gang is used)
    VCC (gang A)    17  (3.3V)      same rail as ADS VDD: the read is ratiometric
    GND (gang A)    25  (GND)
    OTA             -> ADS1115 A0
    gang B          open

The codec sits at 0x1a on the same bus; 0x48 does not collide. Check with
`sudo i2cdetect -y 1` -- want `UU` at 1a (driver bound) and `48` for the ADC.
"""

import argparse
import fcntl
import os
import sys
import time

I2C_SLAVE = 0x0703

REG_CONV = 0x00
REG_CONFIG = 0x01

# MUX for single-ended AINx vs GND, bits [14:12] of the config MSB.
MUX_SINGLE = {0: 0x4, 1: 0x5, 2: 0x6, 3: 0x7}

# PGA full-scale range, bits [11:9]. 0x1 = +/-4.096V, which clears a 3.3V rail
# with headroom to spare. Anything smaller clips the top of the pot travel.
PGA_4V096 = 0x1
FSR_VOLTS = 4.096


class Ads1115:
    def __init__(self, bus=1, addr=0x48):
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)

    def close(self):
        os.close(self.fd)

    def start_continuous(self, channel):
        """Continuous conversion at 128 SPS -- poll the conversion register."""
        msb = (
            0x80                              # OS: start a conversion
            | (MUX_SINGLE[channel] << 4)
            | (PGA_4V096 << 1)
            # MODE bit 0 left clear = continuous
        )
        lsb = 0x80 | 0x03                     # DR=128SPS, comparator disabled
        os.write(self.fd, bytes([REG_CONFIG, msb, lsb]))
        time.sleep(0.01)

    def read(self):
        os.write(self.fd, bytes([REG_CONV]))
        raw = os.read(self.fd, 2)
        val = (raw[0] << 8) | raw[1]
        if val & 0x8000:                      # 16-bit two's complement
            val -= 0x10000
        return val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x48)
    ap.add_argument("--channel", type=int, default=0, choices=(0, 1, 2, 3))
    ap.add_argument("--vref", type=float, default=3.3,
                    help="pot supply, used to normalise position to 0..1")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--once", action="store_true", help="single read, then exit")
    args = ap.parse_args()

    try:
        ads = Ads1115(args.bus, args.addr)
    except FileNotFoundError:
        sys.exit(f"no /dev/i2c-{args.bus} -- is i2c enabled?")
    except OSError as e:
        sys.exit(f"cannot talk to 0x{args.addr:02x} on i2c-{args.bus}: {e}")

    ads.start_continuous(args.channel)

    def sample():
        raw = ads.read()
        volts = raw * FSR_VOLTS / 32768.0
        pos = min(max(volts / args.vref, 0.0), 1.0)
        return raw, volts, pos

    if args.once:
        raw, volts, pos = sample()
        print(f"raw={raw:6d}  {volts:.4f} V  pos={pos:.3f}")
        ads.close()
        return

    lo, hi = 1.0, 0.0
    period = 1.0 / args.hz
    print("slide the pot end to end; ctrl-C to stop")
    try:
        while True:
            raw, volts, pos = sample()
            lo, hi = min(lo, pos), max(hi, pos)
            bar = "#" * int(pos * 40)
            print(f"\rraw={raw:6d}  {volts:.4f} V  pos={pos:.3f}  "
                  f"seen[{lo:.3f},{hi:.3f}]  |{bar:<40}|", end="", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print(f"\ntravel seen: {lo:.3f} .. {hi:.3f}")
    finally:
        ads.close()


if __name__ == "__main__":
    main()
