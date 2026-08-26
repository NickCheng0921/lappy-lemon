"""Measure the GPU-only training-step ceiling, with no dataloader involved.

Every throughput number measured so far was dataloader-bound (ffmpeg decoding
~140ms per chunk), so they say nothing about how fast the 4090 could actually
go. This runs the exact same step -- teacher forward under no_grad + AMP,
student forward/backward/optimiser -- on synthetic tensors, so the only cost is
compute.

The gap between this and the measured 3.76 samples/s is the payoff available
from pre-decoding the dataset. If the gap is small, building the cache is not
worth the disk.

Run: python bench_gpu.py --batch 8 --steps 12
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th

from distill import build_student, spec_l1, stem_l1
from onnx_export import get_htdemucs


def bench(batch, seconds, steps, warmup, amp, arch, dev="cuda"):
    teacher = get_htdemucs().to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build_student(teacher, arch).to(dev).train()

    opt = th.optim.AdamW(student.parameters(), lr=3e-4)
    scaler = th.cuda.amp.GradScaler(enabled=amp)

    n = int(seconds * teacher.samplerate)
    # synthetic stems; values are irrelevant to timing
    stems = th.randn(batch, 4, teacher.audio_channels, n, device=dev) * 0.05

    def step():
        mix = stems.sum(1)
        with th.no_grad(), th.cuda.amp.autocast(enabled=amp):
            t_out = teacher(mix).float()
        with th.cuda.amp.autocast(enabled=amp):
            s_out = student(mix).float()
            loss = 0.5 * stem_l1(s_out, t_out) + 0.5 * stem_l1(s_out, stems)
            loss = loss + 0.1 * spec_l1(s_out, stems)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        th.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    th.cuda.synchronize()
    th.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    th.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps

    peak = th.cuda.max_memory_allocated() / 1e9
    del teacher, student, opt, stems
    th.cuda.empty_cache()
    return dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="4,8")
    ap.add_argument("--seconds", type=float, default=7.8)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    ap.add_argument("--channels", type=int, default=24)
    ap.add_argument("--bottom-channels", type=int, default=256)
    ap.add_argument("--t-layers", type=int, default=4)
    ap.add_argument("--measured", type=float, default=3.76,
                    help="current dataloader-bound samples/s, for comparison")
    args = ap.parse_args()

    arch = dict(channels=args.channels, bottom_channels=args.bottom_channels,
                t_layers=args.t_layers, t_heads=8)
    print(f"GPU: {th.cuda.get_device_name(0)}  amp={args.amp}  arch={arch}")
    print(f"chunk={args.seconds}s  steps={args.steps} (+{args.warmup} warmup)\n")

    hdr = f"{'batch':>6}{'s/step':>9}{'it/s':>8}{'samples/s':>11}{'peak GB':>9}{'vs now':>9}"
    print(hdr)
    print("-" * len(hdr))
    for b in [int(x) for x in args.batches.split(",")]:
        try:
            dt, peak = bench(b, args.seconds, args.steps, args.warmup,
                             args.amp, arch)
        except th.cuda.OutOfMemoryError:
            print(f"{b:>6}{'OOM':>9}")
            th.cuda.empty_cache()
            continue
        sps = b / dt
        print(f"{b:>6}{dt:>9.3f}{1/dt:>8.2f}{sps:>11.2f}{peak:>9.1f}"
              f"{sps/args.measured:>8.2f}x")

    print(f"\n'vs now' = ceiling / your current dataloader-bound "
          f"{args.measured} samples/s.\nThat multiplier is the most a "
          f"pre-decoded cache could buy; anything above it is unreachable.")


if __name__ == "__main__":
    main()
