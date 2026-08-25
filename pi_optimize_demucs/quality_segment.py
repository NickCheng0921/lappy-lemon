"""What does a shorter prediction window cost in separation quality?

htdemucs was trained at 7.8s. Running it at 4s is out-of-distribution: the
cross-transformer sees roughly half the context it learned on. Shrinking the
window only buys ~1.15x speed, so if it also costs real quality it is a bad
trade -- this measures that.

No ground-truth stems needed: we treat the stock 7.8s separation as reference
and measure how far the short-window separation drifts from it. That isolates
the window change from the model's own accuracy.

Run: python quality_segment.py --audio ../profile/calibration_mp3/<file>.mp3 \
                               --seconds 30 --segments 6,5,4,3,2
"""

import argparse
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th

from onnx_export import get_htdemucs


def separate(m, wav, segment, overlap):
    from demucs.apply import apply_model

    m.segment = Fraction(segment).limit_denominator(1000)
    with th.no_grad():
        return apply_model(
            m,
            wav[None],
            shifts=1,
            split=True,
            overlap=overlap,
            device="cpu",
            progress=False,
        )[0]


def snr_db(ref, test):
    noise = th.mean((ref - test) ** 2)
    return float(10 * th.log10(th.mean(ref**2) / (noise + 1e-20)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--segments", default="6,5,4,3,2")
    ap.add_argument("--overlap", type=float, default=0.25)
    args = ap.parse_args()

    from demucs.audio import AudioFile

    m = get_htdemucs()
    ref_seg = float(m.segment)
    sr = m.samplerate
    wav = AudioFile(args.audio).read(
        streams=0, samplerate=sr, channels=m.audio_channels
    )
    wav = wav[:, : int(args.seconds * sr)]
    print(f"{Path(args.audio).name}: {wav.shape[-1]/sr:.1f}s @ {sr} Hz")

    print(f"reference separation at trained segment {ref_seg:.2f}s ...")
    ref = separate(m, wav, ref_seg, args.overlap)

    print(
        f"\n{'seg s':>6}  " + "  ".join(f"{s:>7}" for s in m.sources) + f"{'mean':>9}"
    )
    print("-" * 56)
    for s in [float(x) for x in args.segments.split(",")]:
        out = separate(m, wav, s, args.overlap)
        snrs = [snr_db(ref[i], out[i]) for i in range(len(m.sources))]
        print(
            f"{s:>6.1f}  "
            + "  ".join(f"{v:>7.1f}" for v in snrs)
            + f"{sum(snrs)/len(snrs):>9.1f}"
        )

    print(
        "\nSNR vs the 7.8s reference, in dB. Rough reading: >30 dB is "
        "inaudible drift, ~20 dB is audible on close listening, <15 dB is "
        "a real quality regression."
    )


if __name__ == "__main__":
    main()
