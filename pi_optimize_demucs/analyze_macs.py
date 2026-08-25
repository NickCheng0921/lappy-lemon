"""Per-component MAC breakdown of the htdemucs core.

Optimization strategy depends entirely on WHERE the compute is. Conv and
attention quantize very differently (conv int8 is well supported by ORT's
ARM kernels; attention MatMuls often regress), so we need the split before
picking an approach.

Run:  python pi_optimize_demucs/analyze_macs.py
"""

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th
import torch.nn as nn

from onnx_export import get_htdemucs, make_core_inputs, Core

MACS = defaultdict(int)  # module qualified name -> MACs
COUNTED = []


def _conv_macs(mod, inp, out):
    # MACs = out_elements * (in_channels/groups) * kernel_elements
    k = 1
    for s in mod.kernel_size:
        k *= s
    return out.numel() * (mod.in_channels // mod.groups) * k


def _linear_macs(mod, inp, out):
    return out.numel() * mod.in_features


def _matmul_hook_factory(name):
    """Attention MatMuls aren't nn.Modules; we patch the MHA forward instead."""

    def hook(mod, inp, out):
        pass

    return hook


def register(model):
    handles = []
    for name, mod in model.named_modules():
        if isinstance(
            mod, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
        ):
            fn = _conv_macs
        elif isinstance(mod, nn.Linear):
            fn = _linear_macs
        else:
            continue

        def mk(nm, f):
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                MACS[nm] += f(mod, inp, out)

            return hook

        handles.append(mod.register_forward_hook(mk(name, fn)))
    return handles


def bucket(name):
    """Group module paths into the architectural components."""
    if name.startswith("m.crosstransformer") or name.startswith("crosstransformer"):
        return "crosstransformer"
    for p in ("tencoder", "tdecoder", "encoder", "decoder"):
        if f"{p}." in name or name.startswith(p):
            return p
    return "other"


def main():
    m = get_htdemucs()
    core = Core(m).eval()
    ctx = make_core_inputs(m)
    handles = register(core)
    with th.no_grad():
        core(ctx["x"], ctx["xt"])
    for h in handles:
        h.remove()

    seg = float(m.segment)  # 7.8 s
    groups = defaultdict(int)
    for name, mac in MACS.items():
        groups[bucket(name)] += mac
    total = sum(groups.values())

    print(f"htdemucs core, one {seg:.2f}s window\n")
    print(f"{'component':<20}{'GMAC':>10}{'share':>9}")
    print("-" * 39)
    for g, mac in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{g:<20}{mac/1e9:>10.2f}{100*mac/total:>8.1f}%")
    print("-" * 39)
    print(f"{'TOTAL (hooked)':<20}{total/1e9:>10.2f}{100:>8.1f}%")
    print(f"\nper second of audio (no overlap): {total/1e9/seg:.2f} GMAC/s")
    print(f"with overlap=0.25 (1.43x windows): {1.43*total/1e9/seg:.2f} GMAC/s")

    print("\ntop-15 individual modules:")
    for name, mac in sorted(MACS.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {mac/1e9:8.2f} GMAC  {100*mac/total:5.1f}%  {name}")


if __name__ == "__main__":
    main()
