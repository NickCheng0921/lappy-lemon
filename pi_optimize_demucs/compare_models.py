"""Numerically compare two ONNX cores on identical inputs.

Used to prove a graph transform (simplify, quantize) didn't change behaviour.
For fp32->fp32 transforms expect near-exact agreement; for fp32->int8 expect
real error, which is why --audio-scale matters: judge int8 on realistically
scaled inputs, not on unit-variance noise.

Run:
    python compare_models.py ref.onnx test.onnx [--runs 1]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))


def real_audio_feeds(calib_dir, n):
    """Real-music windows through the torch front end.

    Judging an int8 model on random noise is meaningless: PTQ ranges come from
    music, and N(0,1) noise drives activations far outside them, so everything
    saturates and SNR looks catastrophic even when the model is fine.
    """
    from onnx_export import get_htdemucs, _iter_core_feeds

    m = get_htdemucs()
    return list(_iter_core_feeds(m, calib_dir, n))


def sess(path, threads=0):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument(
        "--audio",
        action="store_true",
        help="use real music windows instead of random noise "
        "(REQUIRED for a meaningful int8 accuracy check)",
    )
    ap.add_argument("--calib-dir", default="profile/calibration_mp3")
    args = ap.parse_args()

    a, b = sess(args.ref), sess(args.test)
    rng = np.random.default_rng(args.seed)

    audio = real_audio_feeds(args.calib_dir, args.runs) if args.audio else None
    if audio:
        print(f"comparing on {len(audio)} real-audio window(s)\n")

    for run in range(args.runs):
        if audio:
            feeds = audio[run]
        else:
            feeds = {}
            for i in a.get_inputs():
                shape = [d if isinstance(d, int) else 1 for d in i.shape]
                feeds[i.name] = rng.standard_normal(shape, dtype=np.float32)

        ra = a.run(None, feeds)
        rb = b.run(None, {i.name: feeds[i.name] for i in b.get_inputs()})

        print(f"run {run}:")
        for out, x, y in zip([o.name for o in a.get_outputs()], ra, rb):
            if x.shape != y.shape:
                print(f"  {out}: SHAPE MISMATCH {x.shape} vs {y.shape}")
                continue
            diff = np.abs(x - y)
            denom = np.abs(x).mean() + 1e-12
            # SNR is the number that tracks audible quality for a masking model.
            noise = np.mean((x - y) ** 2)
            snr = 10 * np.log10(np.mean(x**2) / (noise + 1e-30))
            print(f"  {out:<12} shape={list(x.shape)}")
            print(
                f"     max|d|={diff.max():.3e}  mean|d|={diff.mean():.3e}  "
                f"rel={diff.mean()/denom:.3e}  SNR={snr:.1f} dB"
            )


if __name__ == "__main__":
    main()
