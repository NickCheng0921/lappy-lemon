#!/usr/bin/env python3
"""Read up to 8 pots across two ADS1115s on i2c-1. No deps -- raw /dev/i2c-1.

Five pots needs two boards; the ADS1115 picks its address from the ADDR pin:

    ADDR -> GND   0x48        ADDR -> SDA   0x4a
    ADDR -> VDD   0x49        ADDR -> SCL   0x4b

Both boards share SDA (pin 3), SCL (pin 5), 3.3V (pin 1) and GND (pin 9) --
I2C is a bus, only ADDR differs. Each pot: VCC/GND to the shared 3.3V/GND
rails, wiper to its own Ax.

    board 0x48  A0..A3   pots 1-4
    board 0x49  A0       pot 5      (A1..A3 open)

Unlike single_pot_test.py this cannot use continuous mode: one ADC sits behind
a 4-way mux, so continuous only ever tracks the latched channel. Channels are
round-robined in single-shot instead -- ~7.8ms each at 128 SPS, so a 5-pot scan
is ~30ms (~32 Hz).
"""

import argparse
import fcntl
import os
import sys
import time

I2C_SLAVE = 0x0703

REG_CONV = 0x00
REG_CONFIG = 0x01

MUX_SINGLE = {0: 0x4, 1: 0x5, 2: 0x6, 3: 0x7}

PGA_4V096 = 0x1
FSR_VOLTS = 4.096

# data rate code -> samples/sec, for the conversion-time budget
DATA_RATES = {0: 8, 1: 16, 2: 32, 3: 64, 4: 128, 5: 250, 6: 475, 7: 860}
DR_CODE = 4


class NoBoard(Exception):
    def __init__(self, addr):
        super().__init__(f"no ADS1115 answering at 0x{addr:02x}")
        self.addr = addr


class Ads1115:
    def __init__(self, bus, addr):
        self.addr = addr
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)
        # I2C_SLAVE only latches the target address -- it never touches
        # the bus, so an absent board opens fine and only fails later with
        # EREMOTEIO. Probe here instead, where the error is actionable.
        try:
            os.write(self.fd, bytes([REG_CONFIG]))
            os.read(self.fd, 2)
        except OSError:
            os.close(self.fd)
            raise NoBoard(addr)

    def close(self):
        os.close(self.fd)

    def start_single(self, channel):
        msb = 0x80 | (MUX_SINGLE[channel] << 4) | (PGA_4V096 << 1) | 0x01
        lsb = (DR_CODE << 5) | 0x03
        os.write(self.fd, bytes([REG_CONFIG, msb, lsb]))

    def busy(self):
        """OS bit reads 0 while a conversion is in flight."""
        os.write(self.fd, bytes([REG_CONFIG]))
        return not (os.read(self.fd, 2)[0] & 0x80)

    def read(self):
        os.write(self.fd, bytes([REG_CONV]))
        raw = os.read(self.fd, 2)
        val = (raw[0] << 8) | raw[1]
        if val & 0x8000:
            val -= 0x10000
        return val

    def read_channel(self, channel):
        self.start_single(channel)
        deadline = time.monotonic() + 0.25
        time.sleep(1.0 / DATA_RATES[DR_CODE])
        while self.busy():
            if time.monotonic() > deadline:
                raise TimeoutError(f"0x{self.addr:02x} ch{channel} never finished")
            time.sleep(0.0005)
        return self.read()


def parse_map(spec):
    """"0x48:0,1,2,3 0x49:0" -> [(addr, channel), ...] in pot order."""
    pots = []
    for group in spec.split():
        addr_s, _, chans = group.partition(":")
        addr = int(addr_s, 0)
        for c in chans.split(","):
            pots.append((addr, int(c)))
    return pots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--map", default="0x48:0,1,2,3 0x49:0",
                    help="pot layout, e.g. '0x48:0,1,2,3 0x49:0'")
    ap.add_argument("--names", default="",
                    help="comma-separated labels, e.g. vocals,drums,bass,other,master")
    ap.add_argument("--vref", type=float, default=3.3)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    pots = parse_map(args.map)
    names = [n.strip() for n in args.names.split(",")] if args.names else []
    names += [f"pot{i}" for i in range(len(names), len(pots))]
    width = max(len(n) for n in names[:len(pots)])

    boards = {}
    for addr, _ in pots:
        if addr in boards:
            continue
        try:
            boards[addr] = Ads1115(args.bus, addr)
        except FileNotFoundError:
            sys.exit(f"no /dev/i2c-{args.bus} -- is i2c enabled?")
        except NoBoard as e:
            sys.exit(f"{e} on i2c-{args.bus}\n"
                     f"check ADDR is tied to a rail (GND=0x48 VDD=0x49 "
                     f"SDA=0x4a SCL=0x4b); a floating ADDR is unpredictable\n"
                     f"`sudo i2cdetect -y 1` should list every board")
        except OSError as e:
            sys.exit(f"cannot open 0x{addr:02x} on i2c-{args.bus}: {e}")

    def scan():
        out = []
        for addr, chan in pots:
            raw = boards[addr].read_channel(chan)
            volts = raw * FSR_VOLTS / 32768.0
            out.append((raw, volts, min(max(volts / args.vref, 0.0), 1.0)))
        return out

    if args.once:
        for name, (raw, volts, pos) in zip(names, scan()):
            bar = "#" * int(pos * 30)
            print(f"{name:>{width}}  raw={raw:6d}  {volts:5.3f} V  "
                  f"pos={pos:.3f}  |{bar:<30}|")
        for b in boards.values():
            b.close()
        return

    seen = [(1.0, 0.0)] * len(pots)
    period = 1.0 / args.hz
    print(f"{len(pots)} pots on {len(boards)} board(s); ctrl-C to stop\n")
    try:
        while True:
            vals = scan()
            seen = [(min(lo, p), max(hi, p))
                    for (lo, hi), (_, _, p) in zip(seen, vals)]
            lines = []
            for name, (raw, volts, pos), (lo, hi) in zip(names, vals, seen):
                bar = "#" * int(pos * 30)
                lines.append(f"{name:>{width}}  {volts:5.3f} V  {pos:.3f}  "
                             f"[{lo:.2f},{hi:.2f}]  |{bar:<30}|")
            sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        print("\ntravel seen:")
        for name, (lo, hi) in zip(names, seen):
            print(f"  {name:>{width}}  {lo:.3f} .. {hi:.3f}")
    finally:
        for b in boards.values():
            b.close()


if __name__ == "__main__":
    main()
