"""Export the htdemucs core at an arbitrary window length.

The stock export is locked to the trained 7.8s segment, and htdemucs pads any
shorter input back up to it (htdemucs.py:535), so a short window costs the same
as a long one. Lowering `m.segment` before tracing is what actually produces a
smaller graph.

Quality caveat: htdemucs was TRAINED at 7.8s. Running it at 4s is
out-of-distribution -- the transformer sees less context than it learned on.
Speed here is cheap to measure; the SDR cost is the part that decides whether
it's usable, so evaluate separation quality before adopting.

Run: python export_segment.py --seconds 4 --out models/core_4s.onnx
"""

import argparse
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th

from onnx_export import Core, _patch_sdpa, _patch_unflatten, front, get_htdemucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="htdemucs")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    m = get_htdemucs(args.name)
    print(f"trained segment: {float(m.segment):.2f}s -> exporting at {args.seconds}s")
    m.segment = Fraction(args.seconds).limit_denominator(1000)

    core = Core(m).eval()
    n = int(m.segment * m.samplerate)
    ctx = front(m, th.randn(1, m.audio_channels, n))
    print(f"core inputs: mag={tuple(ctx['x'].shape)} mix_t={tuple(ctx['xt'].shape)}")

    orig_unflatten = _patch_unflatten()
    orig_sdpa = _patch_sdpa()
    # See onnx_export.cmd_export: grad must be ENABLED so MultiheadAttention
    # uses the decomposed (exportable) path rather than the fused kernel.
    with th.enable_grad():
        th.onnx.export(
            core,
            (ctx["x"], ctx["xt"]),
            args.out,
            input_names=["mag", "mix_t"],
            output_names=["spec_out", "time_out"],
            opset_version=args.opset,
            do_constant_folding=True,
        )
    th.Tensor.unflatten = orig_unflatten
    import torch.nn.functional as F

    F.scaled_dot_product_attention = orig_sdpa

    import os

    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
