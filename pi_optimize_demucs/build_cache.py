"""Pre-decode MUSDB stems to a memory-mappable int16 cache.

Measured problem: sampling a chunk from .stem.mp4 costs ~140ms of ffmpeg decode,
which pinned training at 3.76 samples/s against a GPU ceiling of 16.17 -- the
4090 sat idle 77% of the time. Worse, the same audio gets re-decoded hundreds of
times over a 100k-step run.

This decodes each track ONCE into [4 stems, 2 ch, T] int16 .npy. Training then
slices with mmap: no decode, and the OS page cache keeps hot regions resident.

int16 is not a quality compromise -- it is exactly what MUSDB18-HQ ships, and
the source is CD-quality.

Run:
    python build_cache.py --musdb-root ~/data/musdb18 --out ~/data/musdb18_cache
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SOURCES = ["drums", "bass", "other", "vocals"]


def encode_track(job):
    """Decode one track's stems to int16 [4, 2, T]. Runs in a worker process."""
    root, subset, name, is_wav, out_path = job
    import musdb

    if Path(out_path).exists():
        return name, 0, 0.0, True          # already cached

    db = musdb.DB(root=root, subsets=subset, is_wav=is_wav)
    track = next(t for t in db.tracks if t.name == name)
    track.chunk_duration = None             # whole track

    stems = np.stack([track.targets[s].audio.T for s in SOURCES])   # [4,2,T] float
    # MUSDB stems can sit slightly above full scale; clip rather than wrap.
    clipped = int(np.sum(np.abs(stems) > 1.0))
    stems = np.clip(stems, -1.0, 1.0)
    stems16 = (stems * 32767.0).astype(np.int16)

    tmp = str(out_path) + ".tmp.npy"
    np.save(tmp, stems16)
    Path(tmp).replace(out_path)             # atomic: a killed job never leaves
    return name, stems16.shape[-1], clipped / stems.size, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--is-hq", action="store_true")
    ap.add_argument("--subsets", default="train,test")
    ap.add_argument("--jobs", type=int, default=6,
                    help="parallel decoders; match PHYSICAL cores (6 here) -- "
                         "ffmpeg decode is compute-bound so SMT adds little")
    args = ap.parse_args()

    import musdb

    root = str(Path(args.musdb_root).expanduser())
    out = Path(args.out).expanduser()

    jobs = []
    for subset in args.subsets.split(","):
        db = musdb.DB(root=root, subsets=subset, is_wav=args.is_hq)
        if not len(db):
            print(f"!! no tracks in subset {subset!r} under {root}")
            continue
        d = out / subset
        d.mkdir(parents=True, exist_ok=True)
        for t in db.tracks:
            safe = t.name.replace("/", "_")
            jobs.append((root, subset, t.name, args.is_hq, str(d / f"{safe}.npy")))
        print(f"{subset}: {len(db)} tracks -> {d}")

    if not jobs:
        sys.exit("nothing to do")

    print(f"\ndecoding {len(jobs)} track(s) with {args.jobs} workers ...")
    t0 = time.time()
    done = skipped = 0
    total_samples = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for name, n, clip_frac, was_cached in ex.map(encode_track, jobs):
            done += 1
            if was_cached:
                skipped += 1
            else:
                total_samples += n
                if clip_frac > 1e-4:
                    print(f"  ! {name}: {clip_frac*100:.3f}% samples clipped")
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)}  {el/60:.1f} min elapsed", flush=True)

    size = sum(f.stat().st_size for f in out.rglob("*.npy"))
    meta = dict(sources=SOURCES, dtype="int16", scale=32767,
                layout="[stems, channels, samples]",
                tracks=len(jobs), musdb_root=root)
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\ncache: {out}  {size/1e9:.1f} GB  "
          f"({len(jobs)-skipped} decoded, {skipped} already present)")
    print(f"took {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
