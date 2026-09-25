"""Entry point: python -m benchmark.cli --musdb-root /path/to/musdb18hq

Run from the survey_separation/ directory so `benchmark` and `separators` are
both importable.
"""
from __future__ import annotations
import argparse
import os

from benchmark.config import load_musdb_tracks
from benchmark.core.runner import run
from benchmark.core.aggregate import aggregate
from benchmark.core.report import markdown_table
from separators import default_separators


def main():
    ap = argparse.ArgumentParser(description="Stem-separation benchmark")
    ap.add_argument("--musdb-root", default=os.environ.get("MUSDB18_HQ_ROOT"),
                    help="Root of MUSDB18-HQ (or set MUSDB18_HQ_ROOT). Lives outside this tree.")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--run-dir", default="runs/dev")
    ap.add_argument("--profiles", nargs="+", default=["4stem", "2stem"])
    ap.add_argument("--window", type=float, default=1.0,
                    help="BSS Eval window seconds (1.0 = current SDX/SiSEC convention)")
    ap.add_argument("--limit", type=int, default=None, help="First N tracks only")
    ap.add_argument("--spleeter-python", default="python",
                    help="Python interpreter of the Spleeter venv (TF conflicts with torch)")
    args = ap.parse_args()

    if not args.musdb_root:
        raise SystemExit("Set --musdb-root or MUSDB18_HQ_ROOT.")

    tracks = load_musdb_tracks(args.musdb_root, args.subset)
    if args.limit:
        tracks = tracks[: args.limit]
    seps = default_separators(spleeter_python=args.spleeter_python)

    per_track, pending = run(seps, tracks, args.profiles, args.run_dir, win_s=args.window)

    if not per_track.empty:
        summary = aggregate(per_track)
        for p in args.profiles:
            print(f"\n== {p}: median SDR (dB) ==")
            print(markdown_table(summary, p, "sdr"))
    if pending:
        print(f"\n{len(pending)} (separator, track) pairs awaiting manual render "
              f"-> {args.run_dir}/pending.json")


if __name__ == "__main__":
    main()
