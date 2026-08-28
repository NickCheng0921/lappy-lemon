"""How much worse is the model near a window's edges than at its centre?

This decides whether low-latency streaming is viable. Emitting the RIGHT EDGE
of a sliding window removes the whole window-fill term from latency, but the
right edge is exactly where the model has no future context -- which is why
demucs weights window centres and fades the edges.

Method: build a high-quality reference by separating with a dense hop and
keeping only each window's central slice, then compare a SINGLE window's output
against that reference, binned by position within the window. The result tells
you how many seconds of lookahead you must keep to stay near centre quality.

Run: python edge_quality.py --model models/student_final_slim.onnx --bins 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "play_music_on_pi" / "separate"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", default="profile/calibration_mp3/yrshka_sunny.mp3")
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--windows", type=int, default=6,
                    help="how many single-window probes to average over")
    args = ap.parse_args()

    from separate_stream import Separator, read_audio

    sep = Separator(args.model, str(REPO), threads=args.threads)
    win, sr = sep.win, sep.sr
    wav = read_audio(args.audio, sr, sep.ch)

    binw = win // args.bins
    hop = binw                      # dense hop: one bin at a time

    need = win + (args.windows + args.bins) * hop
    if wav.shape[1] < need:
        sys.exit(f"audio too short: need {need/sr:.0f}s, have {wav.shape[1]/sr:.0f}s")
    wav = wav[:, :need]

    # ---- reference: keep only the CENTRAL bin of each window ----------------
    # That slice has maximal context on both sides, so it is the best this
    # model can do at that position.
    centre = (args.bins // 2) * binw
    ref = {}
    n_ref = args.windows + args.bins
    for k in range(n_ref):
        st = k * hop
        stems = sep.separate(wav[:, st : st + win])       # [4, ch, win]
        mixed = stems.sum(0)                              # recombined
        ref[st + centre] = mixed[:, centre : centre + binw].copy()
        if (k + 1) % 5 == 0:
            print(f"  reference {k+1}/{n_ref}", flush=True)

    # ---- probe: one window, scored bin by bin against the reference --------
    snr = np.zeros(args.bins)
    cnt = np.zeros(args.bins)
    for p in range(args.windows):
        st = p * hop
        stems = sep.separate(wav[:, st : st + win])
        mixed = stems.sum(0)
        for b in range(args.bins):
            abs_pos = st + b * binw
            r = ref.get(abs_pos)
            if r is None:
                continue
            t = mixed[:, b * binw : (b + 1) * binw]
            noise = ((r - t) ** 2).mean()
            snr[b] += 10 * np.log10((r**2).mean() / (noise + 1e-20))
            cnt[b] += 1

    ok = cnt > 0
    snr[ok] /= cnt[ok]

    print(f"\nwindow {win/sr:.2f}s split into {args.bins} bins "
          f"of {binw/sr:.2f}s\n")
    print(f"{'bin':>4}{'position in window':>22}{'SNR vs centre':>16}")
    print("-" * 42)
    best = snr[ok].max() if ok.any() else 0.0
    for b in range(args.bins):
        if not ok[b]:
            continue
        a, z = b * binw / sr, (b + 1) * binw / sr
        bar = "#" * max(0, int((snr[b] - best + 20) / 1.5))
        print(f"{b:>4}{f'{a:5.2f}-{z:5.2f}s':>22}{snr[b]:>13.1f} dB  {bar}")

    print("\nHigher = closer to what the model produces with full context.")
    print("The drop at the last bins is the price of emitting the right edge;")
    print("keeping L seconds of lookahead means giving up the worst bins.")


if __name__ == "__main__":
    main()
