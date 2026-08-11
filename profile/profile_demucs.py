"""Profile htdemucs for embedded feasibility: params, memory, FLOPs, wall-clock.

Run with the SAME python/env you use for `python -m demucs`, e.g.:
    python profile/profile_demucs.py
    python profile/profile_demucs.py -n htdemucs --seconds 10 --device cpu

It adds ../vendor to sys.path so the vendored demucs is importable from here.
"""

import argparse
import sys
import time
from pathlib import Path

# --- make the vendored demucs importable regardless of CWD ---
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))

import torch as th
from demucs.pretrained import get_model
from demucs.apply import BagOfModels


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def human(n):
    for unit in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", default="htdemucs")
    ap.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="input length to time a forward pass on. Default: the "
        "model's own training segment (htdemucs requires exactly "
        "this length for a raw forward pass).",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument(
        "--flops",
        action="store_true",
        help="also estimate FLOPs via thop (pip install thop)",
    )
    args = ap.parse_args()

    print(f"Loading model '{args.name}' ...")
    model = get_model(args.name)
    model.eval()

    # A bag (like htdemucs) is several submodels; report each + total.
    submodels = model.models if isinstance(model, BagOfModels) else [model]
    sr = model.samplerate
    ch = model.audio_channels
    print(
        f"sources={model.sources}  samplerate={sr}  channels={ch}  "
        f"submodels={len(submodels)}"
    )

    total = 0
    for i, m in enumerate(submodels):
        p = count_params(m)
        total += p
        seg = getattr(m, "segment", "n/a")
        print(f"  submodel[{i}] params={human(p)}  segment={seg}")
    print(
        f"TOTAL params={human(total)}  weight memory(fp32)={human(total*4)}B  "
        f"(int8={human(total)}B)"
    )

    # --- timing / peak activation memory on a dummy input ---
    dev = args.device
    if dev == "cuda" and not th.cuda.is_available():
        print("cuda not available, falling back to cpu")
        dev = "cpu"
    # htdemucs requires a raw forward to be EXACTLY its training segment.
    # Default to that length; only override if the user explicitly asks.
    m0 = submodels[0].to(dev)
    seconds = args.seconds if args.seconds is not None else float(m0.segment)
    n = int(seconds * sr)
    x = th.randn(1, ch, n, device=dev)

    print(
        f"\nTiming one submodel forward on {seconds:.3f}s @ {dev} "
        f"(input {tuple(x.shape)}) ..."
    )
    with th.no_grad():
        m0(x)  # warmup
        if dev == "cuda":
            th.cuda.reset_peak_memory_stats()
            th.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.runs):
            m0(x)
        if dev == "cuda":
            th.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.runs

    rtf = dt / seconds
    print(
        f"  per-forward: {dt*1000:.1f} ms  |  real-time factor: {rtf:.2f}x "
        f"({'faster' if rtf < 1 else 'SLOWER'} than real-time, one submodel)"
    )
    print(
        f"  full bag (~{len(submodels)}x) est: {dt*len(submodels)*1000:.1f} ms "
        f"-> RTF {rtf*len(submodels):.2f}x"
    )
    if dev == "cuda":
        print(
            f"  peak CUDA activation mem: " f"{human(th.cuda.max_memory_allocated())}B"
        )

    if args.flops:
        try:
            from thop import profile as thop_profile

            macs, _ = thop_profile(m0, inputs=(x,), verbose=False)
            print(
                f"\nFLOPs (thop, one submodel, {seconds:.3f}s): "
                f"{human(macs*2)}  (MACs={human(macs)})"
            )
        except ImportError:
            print("\n(thop not installed; `pip install thop` for FLOPs)")


if __name__ == "__main__":
    main()
