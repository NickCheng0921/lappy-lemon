"""Print the concrete conv / FFN dimensions of the pretrained htdemucs core,
with the measured per-layer MACs, and show the execution ordering.

Also records each layer's activation output shape, since the Pi is
bandwidth-bound and activation bytes matter as much as MACs.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th
import torch.nn as nn

from onnx_export import Core, get_htdemucs, make_core_inputs

ROWS = []


def hook(name, mod):
    def h(m, i, o):
        o = o[0] if isinstance(o, tuple) else o
        if isinstance(m, nn.Linear):
            mac = o.numel() * m.in_features
            desc = f"Linear {m.in_features}->{m.out_features}"
        else:
            k = 1
            for s in m.kernel_size:
                k *= s
            mac = o.numel() * (m.in_channels // m.groups) * k
            desc = (
                f"{type(m).__name__[:13]} {m.in_channels}->{m.out_channels} "
                f"k={tuple(m.kernel_size)} s={tuple(m.stride)}"
            )
        ROWS.append((name, desc, mac, tuple(o.shape), o.numel() * 4 / 1e6))

    return h


def main():
    m = get_htdemucs()
    core = Core(m).eval()
    ctx = make_core_inputs(m)

    handles = []
    for name, mod in core.named_modules():
        if isinstance(
            mod,
            (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.Linear),
        ):
            handles.append(mod.register_forward_hook(hook(name, mod)))
    with th.no_grad():
        core(ctx["x"], ctx["xt"])
    for h in handles:
        h.remove()

    print(f"{'#':>3} {'layer':<44}{'shape':<34}{'GMAC':>7}{'actMB':>8}")
    print("-" * 96)
    tot = act = 0
    for i, (name, desc, mac, shape, mb) in enumerate(ROWS):
        tot += mac
        act += mb
        short = name.replace("m.", "")
        print(
            f"{i:>3} {short[:26]:<27}{desc[:16]:<17}{str(shape):<34}"
            f"{mac/1e9:>7.2f}{mb:>8.1f}"
        )
    print("-" * 96)
    print(f"{'TOTAL':<48}{'':<34}{tot/1e9:>7.1f}{act:>8.0f}")
    print(
        f"\ntotal activation bytes written by these layers: {act/1000:.1f} GB "
        f"per window (this is what makes the Pi bandwidth-bound)"
    )


if __name__ == "__main__":
    main()
