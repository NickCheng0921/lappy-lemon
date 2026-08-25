"""How does cost scale with the prediction window?

htdemucs pads any short input back up to `training_length` (htdemucs.py:535),
so feeding 4s of audio to the stock model costs exactly as much as 7.8s. To
actually shrink the window you must lower `m.segment`, which is what this does.

Two competing effects, which is why this needs measuring rather than guessing:
  * conv cost is LINEAR in window length -> cost per second of audio is flat,
    except that fixed padding/edge overhead grows as windows get shorter.
  * attention is QUADRATIC in sequence length -> halving the window quarters
    the attention cost, i.e. per second of audio it HALVES. This is the win.
  * but overlap-add means shorter windows need MORE windows per second:
    windows/s = 1 / (seg * (1 - overlap)).

Reports MACs per second of audio, which is the number that decides RTF.

Run: python sweep_segment.py [--overlap 0.25]
"""

import argparse
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th
import torch.nn as nn

from onnx_export import Core, front, get_htdemucs


def hook_macs(core):
    """Count Conv/Linear MACs, and attention matmul MACs (which are not
    nn.Modules and would otherwise be invisible -- they are the quadratic part)."""
    macs = defaultdict(int)
    seqlens = []
    handles = []

    for name, mod in core.named_modules():
        if isinstance(
            mod, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
        ):
            k = 1
            for s in mod.kernel_size:
                k *= s

            def mk(nm, mod=mod, k=k):
                def h(m, i, o):
                    o = o[0] if isinstance(o, tuple) else o
                    macs["conv"] += o.numel() * (m.in_channels // m.groups) * k

                return h

            handles.append(mod.register_forward_hook(mk(name)))
        elif isinstance(mod, nn.Linear):

            def mkl(nm, mod=mod):
                def h(m, i, o):
                    o = o[0] if isinstance(o, tuple) else o
                    macs["linear"] += o.numel() * m.in_features

                return h

            handles.append(mod.register_forward_hook(mkl(name)))
        elif isinstance(mod, nn.MultiheadAttention):

            def mka(nm, mod=mod):
                def h(m, i, o):
                    # i[0] is query: (L, N, E) or (N, L, E) depending on batch_first
                    q = i[0]
                    L = q.shape[1] if getattr(m, "batch_first", False) else q.shape[0]
                    E = q.shape[-1]
                    seqlens.append(L)
                    # QK^T and AV: each L*L*E MACs
                    macs["attn_matmul"] += 2 * L * L * E

                return h

            handles.append(mod.register_forward_hook(mka(name)))

    return macs, seqlens, handles


def measure(seconds, overlap):
    m = get_htdemucs()
    m.segment = Fraction(seconds).limit_denominator(1000)
    core = Core(m).eval()

    sr = m.samplerate
    n = int(m.segment * sr)
    mix = th.randn(1, m.audio_channels, n)
    ctx = front(m, mix)

    macs, seqlens, handles = hook_macs(core)
    with th.no_grad():
        core(ctx["x"], ctx["xt"])
    for h in handles:
        h.remove()

    total = sum(macs.values())
    # windows needed per second of audio under overlap-add
    per_sec = total / (seconds * (1.0 - overlap))
    return dict(
        seconds=seconds,
        spec_shape=tuple(ctx["x"].shape),
        conv=macs["conv"],
        linear=macs["linear"],
        attn=macs["attn_matmul"],
        total=total,
        per_sec=per_sec,
        seqlen=max(seqlens) if seqlens else 0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--segments", default="1,2,3,4,5,6,7.8")
    args = ap.parse_args()

    segs = [float(s) for s in args.segments.split(",")]
    rows = []
    for s in segs:
        try:
            rows.append(measure(s, args.overlap))
        except Exception as e:
            print(f"  {s}s FAILED: {type(e).__name__}: {e}")

    base = next((r for r in rows if abs(r["seconds"] - 7.8) < 1e-6), rows[-1])

    print(f"\noverlap={args.overlap}  (windows/s = 1/(seg*{1-args.overlap:.2f}))\n")
    hdr = (
        f"{'seg s':>6}{'specT':>7}{'attn L':>8}{'conv G':>9}{'lin G':>8}"
        f"{'attn G':>8}{'tot G':>8}{'GMAC/s':>9}{'vs 7.8s':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['seconds']:>6.1f}{r['spec_shape'][3]:>7}{r['seqlen']:>8}"
            f"{r['conv']/1e9:>9.1f}{r['linear']/1e9:>8.1f}{r['attn']/1e9:>8.1f}"
            f"{r['total']/1e9:>8.1f}{r['per_sec']/1e9:>9.2f}"
            f"{base['per_sec']/r['per_sec']:>8.2f}x"
        )

    print(
        "\nRTF projection from the measured Pi conv-int8 run "
        "(7.8s window = 13.63s wall, core RTF 1.75x):"
    )
    for r in rows:
        # scale the measured 7.8s wall time by relative MACs/s of audio
        proj = 1.747 * (r["per_sec"] / base["per_sec"])
        print(
            f"  seg={r['seconds']:>4.1f}s -> projected core RTF {proj:.2f}x"
            + ("   <-- target 0.8x" if proj <= 0.8 else "")
        )
    print(
        "\nNB: projection assumes time scales with MACs. The Pi is "
        "bandwidth-bound, so shorter windows should do BETTER than this "
        "(smaller activations), but quality must be checked separately."
    )


if __name__ == "__main__":
    main()
