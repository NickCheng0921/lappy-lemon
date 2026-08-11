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
from demucs.apply import BagOfModels, apply_model


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def peak_rss_mb():
    """Process-wide peak resident memory (high-water mark).

    Linux ru_maxrss is in KB, macOS in bytes. Returns MB, or None on Windows
    (no `resource` module there -- run under WSL/Linux for this metric).
    """
    try:
        import resource
    except ImportError:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def fmt_mb(mb):
    return "n/a (Windows: use WSL)" if mb is None else f"{mb:.0f} MB"


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
    ap.add_argument(
        "--track-seconds",
        type=float,
        default=None,
        help="run the FULL apply_model pipeline (split + overlap + shifts) on a "
        "dummy track of this length. Gives aggregate GMACs, per-second-of-audio "
        "compute, real-time factor, and peak RAM for a realistic multi-window run.",
    )
    ap.add_argument("--shifts", type=int, default=1)
    ap.add_argument("--overlap", type=float, default=0.25)
    args = ap.parse_args()

    print(f"peak RSS at start: {fmt_mb(peak_rss_mb())}")

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
    print(f"peak RSS after load: {fmt_mb(peak_rss_mb())}")

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

    print(f"  peak RSS after single forward: {fmt_mb(peak_rss_mb())}")

    per_window_macs = None
    if args.flops:
        try:
            from thop import profile as thop_profile

            per_window_macs, _ = thop_profile(m0, inputs=(x,), verbose=False)
            print(
                f"\nFLOPs (thop, one submodel, {seconds:.3f}s): "
                f"{human(per_window_macs*2)}  (MACs={human(per_window_macs)})"
            )
        except ImportError:
            print("\n(thop not installed; `pip install thop` for FLOPs)")

    # --- full pipeline on a longer dummy track: aggregate GMACs + peak RAM ---
    if args.track_seconds:
        length = int(args.track_seconds * sr)
        track = th.randn(1, ch, length, device=dev)

        # window geometry that apply_model will use (matches apply.py)
        seg_len = int(sr * float(m0.segment))
        stride = int((1 - args.overlap) * seg_len)
        n_windows = len(range(0, length, stride))
        passes = max(args.shifts, 1)  # shifts>=1 -> that many full sweeps
        total_windows = n_windows * passes * len(submodels)

        print(
            f"\n=== full apply_model on {args.track_seconds:.1f}s track "
            f"(shifts={args.shifts}, overlap={args.overlap}) ==="
        )
        print(
            f"  window={float(m0.segment):.2f}s  stride={stride/sr:.2f}s  "
            f"windows/pass={n_windows}  passes={passes}  "
            f"submodels={len(submodels)}  -> total window-forwards={total_windows}"
        )

        with th.no_grad():
            t0 = time.perf_counter()
            apply_model(
                model,
                track,
                shifts=args.shifts,
                split=True,
                overlap=args.overlap,
                device=dev,
                progress=False,
            )
            dt_full = time.perf_counter() - t0

        rtf_full = dt_full / args.track_seconds
        print(
            f"  wall-clock: {dt_full:.1f}s for {args.track_seconds:.1f}s audio "
            f"-> RTF {rtf_full:.2f}x "
            f"({'faster' if rtf_full < 1 else 'SLOWER'} than real-time)"
        )
        print(f"  peak RSS during full run: {fmt_mb(peak_rss_mb())}")
        # the full-track output buffer apply_model holds in RAM:
        out_buf_mb = (len(model.sources) * ch * length * 4) / (1024 * 1024)
        print(
            f"  full-track output buffer (sources*ch*len*fp32): {out_buf_mb:.0f} MB "
            f"(grows with track length -- a streaming impl would NOT hold this)"
        )

        if per_window_macs is not None:
            agg_macs = per_window_macs * total_windows
            per_sec = agg_macs / args.track_seconds
            print(
                f"  aggregate compute: {human(agg_macs)}MAC total  |  "
                f"{human(per_sec)}MAC per second of audio "
                f"(divide a device's MAC/s into this for a real-time verdict)"
            )
        else:
            print("  (add --flops to also get aggregate GMAC numbers)")


if __name__ == "__main__":
    main()
