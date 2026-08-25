"""Export a randomly-initialised student core, purely to measure its Pi speed.

Weights are random: timing does not depend on values, and this runs BEFORE any
training so we can kill the idea cheaply if the architecture is still too slow.

Run: python export_student.py --channels 32 --bottom-channels 256 --t-layers 4 \
                              --seconds 7.8 --out models/student.onnx
"""

import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th

from demucs.htdemucs import HTDemucs
from onnx_export import Core, _patch_sdpa, _patch_unflatten, front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--bottom-channels", type=int, default=256)
    ap.add_argument("--t-layers", type=int, default=4)
    ap.add_argument("--t-heads", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=39 / 5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ckpt", default="",
                    help="load trained weights from a prune_init/distill .pt; "
                         "its stored arch overrides the --channels flags so the "
                         "exported graph always matches the checkpoint")
    args = ap.parse_args()

    ckpt = None
    if args.ckpt:
        ckpt = th.load(args.ckpt, map_location="cpu")
        a = ckpt["arch"]
        args.channels = a["channels"]
        args.bottom_channels = a["bottom_channels"]
        args.t_layers = a["t_layers"]
        args.t_heads = a.get("t_heads", args.t_heads)
        print(f"loaded {args.ckpt} (step {ckpt.get('step', 'n/a')}) arch={a}")

    m = HTDemucs(
        sources=["drums", "bass", "other", "vocals"],
        channels=args.channels,
        bottom_channels=args.bottom_channels,
        t_layers=args.t_layers,
        t_heads=args.t_heads,
        segment=Fraction(args.seconds).limit_denominator(1000),
    ).eval()
    if ckpt is not None:
        m.load_state_dict(ckpt["state_dict"])
        print("weights loaded into student")
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(
        f"student: channels={args.channels} bottom={args.bottom_channels} "
        f"t_layers={args.t_layers} -> {params:.1f}M params, "
        f"segment={float(m.segment):.2f}s"
    )

    core = Core(m).eval()
    ctx = front(m, th.randn(1, m.audio_channels, int(m.segment * m.samplerate)))
    print(f"core inputs: mag={tuple(ctx['x'].shape)} mix_t={tuple(ctx['xt'].shape)}")

    orig_unflatten = _patch_unflatten()
    orig_sdpa = _patch_sdpa()
    with th.enable_grad():  # required for exportable (decomposed) attention
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
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
