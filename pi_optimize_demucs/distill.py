"""Distill the htdemucs teacher into a pruned student on MUSDB18 (+ optional
unlabeled audio).

Loss = a*L1(student, teacher_output) + (1-a)*L1(student, ground_truth_stems)

Why both terms:
  * the teacher term needs NO stems, so it can use unlimited unlabeled music --
    the main reason to distill rather than train small from scratch. MUSDB18 is
    only ~10 h, which is thin for a 8.8M-param model.
  * the ground-truth term stops the student inheriting the teacher's mistakes.
    Pure distillation caps you at the teacher and usually below it.

Data sources (either or both):
  --musdb-root   MUSDB18 (stem .mp4) or MUSDB18-HQ (wav dirs) -> supervised + distill
  --unlabeled    folder of any audio                          -> distill only

Run a smoke test first (downloads musdb's tiny built-in sample):
    python distill.py --smoke
Real run:
    python distill.py --musdb-root /data/musdb18hq --is-hq \
        --init models/student_pruned.pt --steps 200000 --batch 4
"""

import argparse
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from demucs.htdemucs import HTDemucs
from onnx_export import get_htdemucs

SOURCES = ["drums", "bass", "other", "vocals"]


# ----------------------------------------------------------------------- data
class MusdbChunks(th.utils.data.Dataset):
    """Random fixed-length chunks with stems. Length is virtual -- an 'epoch'
    is just `samples_per_epoch` random draws, which is how demucs trains."""

    def __init__(
        self, root, seconds, sr, is_hq, subset="train", samples_per_epoch=4000
    ):
        import musdb

        self.db = musdb.DB(root=root, subsets=subset, is_wav=is_hq)
        if not len(self.db):
            raise RuntimeError(
                f"no MUSDB tracks under {root!r} "
                f"(is_wav={is_hq}); check --musdb-root/--is-hq"
            )
        self.n = int(seconds * sr)
        self.sr = sr
        self.len = samples_per_epoch

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        tr = random.choice(self.db.tracks)
        tr.chunk_duration = self.n / self.sr
        # random start; musdb resamples internally to its own rate
        dur = max(tr.duration - tr.chunk_duration, 0.0)
        tr.chunk_start = random.uniform(0, dur)
        stems = th.stack(
            [th.tensor(tr.targets[s].audio.T, dtype=th.float32) for s in SOURCES]
        )  # [S, C, T]
        stems = self._fit(stems)
        return stems.sum(0), stems  # mix, stems

    def _fit(self, x):
        if x.shape[-1] < self.n:
            x = F.pad(x, (0, self.n - x.shape[-1]))
        return x[..., : self.n]


class AudioChunks(th.utils.data.Dataset):
    """Unlabeled audio -> mixture only. Used for the distillation term."""

    def __init__(self, folder, seconds, sr, channels=2, samples_per_epoch=4000):
        from demucs.audio import AudioFile

        exts = ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg")
        self.files = sorted(p for e in exts for p in Path(folder).rglob(e))
        if not self.files:
            raise RuntimeError(f"no audio under {folder!r}")
        self.AudioFile = AudioFile
        self.n = int(seconds * sr)
        self.sr, self.ch, self.len = sr, channels, samples_per_epoch
        print(f"unlabeled: {len(self.files)} file(s)")

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        for _try in range(8):
            p = random.choice(self.files)
            try:
                wav = self.AudioFile(p).read(
                    streams=0, samplerate=self.sr, channels=self.ch
                )
            except Exception:
                continue
            if wav.shape[-1] <= self.n:
                return F.pad(wav, (0, self.n - wav.shape[-1])), None
            s = random.randint(0, wav.shape[-1] - self.n - 1)
            return wav[..., s : s + self.n], None
        raise RuntimeError("could not read any audio file")


def augment(stems):
    """demucs-style augmentation: random per-source gain/sign, channel swap,
    and cross-track source shuffling within the batch (a strong regulariser
    that effectively multiplies the tiny MUSDB training set)."""
    B, S, C, T = stems.shape
    # shuffle each source independently across the batch
    for s in range(S):
        stems[:, s] = stems[th.randperm(B), s]
    gain = th.empty(B, S, 1, 1, device=stems.device).uniform_(0.25, 1.25)
    sign = th.where(th.rand(B, S, 1, 1, device=stems.device) < 0.5, -1.0, 1.0)
    stems = stems * gain * sign
    if random.random() < 0.5:  # swap L/R
        stems = stems.flip(2)
    return stems


# ---------------------------------------------------------------------- model
def build_student(teacher, arch):
    return HTDemucs(
        sources=list(teacher.sources),
        audio_channels=teacher.audio_channels,
        channels=arch["channels"],
        bottom_channels=arch["bottom_channels"],
        t_layers=arch["t_layers"],
        t_heads=arch.get("t_heads", 8),
        nfft=teacher.nfft,
        depth=teacher.depth,
        segment=teacher.segment,
    )


def load_student(teacher, init_path, arch_override):
    if init_path and Path(init_path).exists():
        ck = th.load(init_path, map_location="cpu")
        arch = ck["arch"]
        arch.update({k: v for k, v in arch_override.items() if v is not None})
        st = build_student(teacher, arch)
        st.load_state_dict(ck["state_dict"])
        print(f"student initialised from pruned teacher: {init_path}")
        return st, arch
    arch = {k: v for k, v in arch_override.items()}
    print("student randomly initialised (no --init given)")
    return build_student(teacher, arch), arch


# ----------------------------------------------------------------------- loss
def stem_l1(pred, target):
    return F.l1_loss(pred, target)


def spec_l1(pred, target, n_fft=2048):
    """Multi-resolution magnitude term. Waveform L1 alone tolerates phase-y
    smearing that is clearly audible; a magnitude term penalises it.

    NORMALISED by the target's mean magnitude. Un-normalised, STFT magnitudes
    run ~15x the waveform L1 (measured: spec=83 vs teach=5.7), so the spec term
    silently swamps the other two and spec_weight stops meaning anything.
    """
    loss = 0.0
    for nf in (n_fft, n_fft // 2, n_fft // 4):
        w = th.hann_window(nf, device=pred.device)
        f = lambda x: th.stft(
            x.reshape(-1, x.shape[-1]), nf, nf // 4, window=w, return_complex=True
        ).abs()
        tm = f(target)
        loss = loss + F.l1_loss(f(pred), tm) / (tm.mean() + 1e-8)
    return loss / 3.0


# ---------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb-root", default="")
    ap.add_argument("--is-hq", action="store_true", help="MUSDB18-HQ (wav dirs)")
    ap.add_argument("--unlabeled", default="", help="folder of extra audio")
    ap.add_argument("--init", default="pi_optimize_demucs/models/student_pruned.pt")
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--bottom-channels", type=int, default=None)
    ap.add_argument("--t-layers", type=int, default=None)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seconds", type=float, default=7.8)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="weight on the TEACHER-matching term; 1.0 = pure "
        "distillation, 0.0 = pure supervised",
    )
    ap.add_argument("--spec-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--out", default="pi_optimize_demucs/models/student_distilled.pt")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run on musdb's built-in sample to prove the "
        "pipeline works end to end",
    )
    args = ap.parse_args()

    dev = "cuda" if th.cuda.is_available() else "cpu"
    if args.smoke:
        args.steps, args.batch, args.seconds = 6, 1, 2.0
        args.workers, args.save_every, args.log_every = 0, 10_000, 1

    teacher = get_htdemucs().to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student, arch = load_student(
        teacher,
        args.init,
        dict(
            channels=args.channels,
            bottom_channels=args.bottom_channels,
            t_layers=args.t_layers,
        ),
    )
    student = student.to(dev).train()
    sp = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"student {sp:.1f}M params  arch={arch}  device={dev}")

    # ---- data
    sr = teacher.samplerate
    sets = []
    if args.smoke:
        import musdb

        db = musdb.DB(download=True, subsets="train")
        print(f"smoke: musdb sample has {len(db)} track(s)")
        ds = MusdbChunks.__new__(MusdbChunks)
        ds.db, ds.n, ds.sr, ds.len = db, int(args.seconds * sr), sr, args.steps
        sets.append(ds)
    else:
        if args.musdb_root:
            sets.append(MusdbChunks(args.musdb_root, args.seconds, sr, args.is_hq))
        if args.unlabeled:
            sets.append(
                AudioChunks(args.unlabeled, args.seconds, sr, teacher.audio_channels)
            )
    if not sets:
        raise SystemExit("give --musdb-root and/or --unlabeled (or --smoke)")

    def collate(batch):
        mixes = th.stack([b[0] for b in batch])
        have = all(b[1] is not None for b in batch)
        stems = th.stack([b[1] for b in batch]) if have else None
        return mixes, stems

    loaders = [
        th.utils.data.DataLoader(
            d,
            batch_size=args.batch,
            num_workers=args.workers,
            collate_fn=collate,
            drop_last=True,
        )
        for d in sets
    ]

    opt = th.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = th.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )
    scaler = th.cuda.amp.GradScaler(enabled=args.amp and dev == "cuda")

    step, t0 = 0, time.time()
    running = {}
    print(
        f"\ntraining {args.steps} steps, batch {args.batch}, "
        f"{args.seconds}s chunks, alpha={args.alpha}\n"
    )

    while step < args.steps:
        for loader in loaders:
            for mix, stems in loader:
                if step >= args.steps:
                    break
                mix = mix.to(dev, non_blocking=True)
                if stems is not None:
                    stems = stems.to(dev, non_blocking=True)
                    stems = augment(stems)
                    mix = stems.sum(1)

                with th.no_grad(), th.cuda.amp.autocast(
                    enabled=args.amp and dev == "cuda"
                ):
                    t_out = teacher(mix).float()

                with th.cuda.amp.autocast(enabled=args.amp and dev == "cuda"):
                    s_out = student(mix)
                    s_out = s_out.float()
                    l_teach = stem_l1(s_out, t_out)
                    loss = args.alpha * l_teach
                    parts = {"teach": l_teach.item()}
                    if stems is not None:
                        l_sup = stem_l1(s_out, stems)
                        loss = loss + (1 - args.alpha) * l_sup
                        parts["sup"] = l_sup.item()
                    if args.spec_weight:
                        ref = stems if stems is not None else t_out
                        l_spec = spec_l1(s_out, ref)
                        loss = loss + args.spec_weight * l_spec
                        parts["spec"] = l_spec.item()

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                th.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

                parts["loss"] = loss.item()
                for k, v in parts.items():
                    running[k] = running.get(k, 0.0) + v
                step += 1

                bar.update(1)
                if step % max(1, args.log_every // 5) == 0:
                    bar.set_postfix(loss=f"{parts['loss']:.3f}",
                                    teach=f"{parts['teach']:.3f}",
                                    lr=f"{sched.get_last_lr()[0]:.1e}")

                if step % args.log_every == 0:
                    msg = "  ".join(
                        f"{k}={v/args.log_every:.4f}" for k, v in running.items()
                    )
                    rate = step / (time.time() - t0)
                    print(
                        f"step {step:>7}/{args.steps}  {msg}  "
                        f"lr={sched.get_last_lr()[0]:.2e}  {rate:.2f} it/s",
                        flush=True,
                    )
                    running = {}

                if step % args.save_every == 0:
                    save(student, arch, args.out, step)

    save(student, arch, args.out, step)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


def save(student, arch, out, step):
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    th.save({"state_dict": student.state_dict(), "arch": arch, "step": step}, p)
    print(f"  saved {p} @ step {step}")


if __name__ == "__main__":
    main()
