"""Standalone RTF benchmark for an exported htdemucs core .onnx.

Deliberately depends on ONLY onnxruntime + numpy -- no torch, no demucs. The
core is ~all of htdemucs' compute (130 GMAC of the ~130 GMAC total), so its
RTF is the number that matters, and dropping torch means the Pi needs a 30-second
pip install instead of a multi-GB one.

Inputs are random, which is fine for TIMING (int8 kernel speed doesn't depend on
values). Use onnx_export.py --verify for ACCURACY -- never judge quality here.

Run:
    python bench_core.py MODEL.onnx [--threads 1] [--runs 5] [--lean]
"""

import argparse
import json
import os
import resource
import time

import numpy as np
import onnxruntime as ort

WINDOW_SECONDS = 39 / 5  # htdemucs trained segment = 7.8s


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def make_session(path, threads, lean, opt_level="all", profile=False):
    so = ort.SessionOptions()
    if profile:
        so.enable_profiling = True
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[opt_level]
    if lean:
        # 4GB Pi: the arena allocator roughly doubles peak RSS on this graph.
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def summarize_profile(prof_path):
    """Aggregate ORT's per-node trace by op type.

    The key question for a QDQ int8 model: did Conv fuse into QLinearConv?
    If you see fp32 'Conv' plus heavy 'DequantizeLinear'/'QuantizeLinear',
    the quantization is not paying off -- it's costing extra.
    """
    from collections import defaultdict

    with open(prof_path) as fh:
        events = json.load(fh)

    by_type = defaultdict(lambda: [0.0, 0])  # op_type -> [total_us, count]
    by_node = defaultdict(float)
    total = 0.0
    for e in events:
        if e.get("cat") != "Node" or "dur" not in e:
            continue
        arg = e.get("args", {})
        op = arg.get("op_name")
        if not op:
            continue
        dur = e["dur"]
        by_type[op][0] += dur
        by_type[op][1] += 1
        by_node[e.get("name", "?")] += dur
        total += dur

    print(f"\n--- ORT op profile ({os.path.basename(prof_path)}) ---")
    print(f"{'op type':<26}{'ms':>10}{'share':>8}{'count':>8}")
    print("-" * 52)
    for op, (us, n) in sorted(by_type.items(), key=lambda kv: -kv[1][0])[:18]:
        print(f"{op:<26}{us/1000:>10.1f}{100*us/total:>7.1f}%{n:>8}")
    print("-" * 52)
    print(f"{'TOTAL':<26}{total/1000:>10.1f}")

    print("\ntop-10 slowest individual nodes:")
    for name, us in sorted(by_node.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {us/1000:9.1f} ms  {100*us/total:5.1f}%  {name[:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--opt-level", default="all")
    ap.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    ap.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="only used to project a track-level RTF",
    )
    ap.add_argument("--json", default="", help="append one JSON result line here")
    ap.add_argument("--tag", default="", help="label for the JSON row")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="ORT per-op profiling; prints time by op type. Shows "
        "whether Conv actually ran as int8 QLinearConv or fell back "
        "to fp32 with Quantize/Dequantize overhead around it.",
    )
    args = ap.parse_args()

    print(f"model: {args.model}  ({os.path.getsize(args.model)/1e6:.0f} MB)")
    print(f"threads={args.threads} lean={args.lean} opt={args.opt_level}")

    sess = make_session(
        args.model, args.threads, args.lean, args.opt_level, profile=args.profile
    )
    print(f"peak RSS after session init: {peak_rss_mb():.0f} MB")

    rng = np.random.default_rng(0)
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        feeds[i.name] = rng.standard_normal(shape, dtype=np.float32)
        print(f"  input {i.name}: {shape}")

    for _ in range(args.warmup):
        sess.run(None, feeds)
    print(f"peak RSS after warmup forward: {peak_rss_mb():.0f} MB")

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append(time.perf_counter() - t0)

    times.sort()
    best, med = times[0], times[len(times) // 2]
    w = args.window_seconds
    # Track-level RTF: overlap makes windows overlap, so a track needs
    # 1/(1-overlap) more window-forwards per second of audio.
    track_factor = 1.0 / (1.0 - args.overlap)

    print(
        f"\nper-window forward over {args.runs} runs (s): "
        f"best={best:.2f} median={med:.2f} worst={times[-1]:.2f}"
    )
    print(f"  window = {w:.2f}s audio")
    print(f"  CORE RTF (no overlap)      : {med/w:.2f}x")
    print(f"  proj. track RTF @ overlap={args.overlap}: " f"{med/w*track_factor:.2f}x")
    print(f"peak RSS overall: {peak_rss_mb():.0f} MB")

    if args.profile:
        summarize_profile(sess.end_profiling())

    if args.json:
        row = dict(
            tag=args.tag or os.path.basename(args.model),
            model=os.path.basename(args.model),
            size_mb=round(os.path.getsize(args.model) / 1e6, 1),
            threads=args.threads,
            lean=args.lean,
            opt=args.opt_level,
            runs=args.runs,
            best_s=round(best, 3),
            median_s=round(med, 3),
            core_rtf=round(med / w, 3),
            track_rtf=round(med / w * track_factor, 3),
            peak_rss_mb=round(peak_rss_mb()),
        )
        with open(args.json, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"appended to {args.json}")


if __name__ == "__main__":
    main()
