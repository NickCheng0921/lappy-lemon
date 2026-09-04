"""Regression gate for the streaming app. Run after ANY edit.

The structural checks need no model and no audio device, so they are cheap
enough to run on every change. --model adds the end-to-end offline check.

    python selftest.py                      # structural only
    python selftest.py --model M.onnx --audio clip.wav --repo ..
"""

import argparse
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audio_io
import gains as G
from separator import Separator
from stream import Stream

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def fake_sep(ch=1, win=343980, sr=44100):
    """Identity 'model': stem 0 is the input, the rest silent, so the four
    stems sum back to exactly the input. Any assembly error shows up as error."""
    f = types.SimpleNamespace(sr=sr, ch=ch, win=win)
    f.separate = lambda w: np.stack([w, np.zeros_like(w), np.zeros_like(w),
                                     np.zeros_like(w)])
    return f


def run_stream(sep, sig, chunk, **kw):
    st = Stream(sep, dict(G.GAINS), kw.pop("stride", 4.0), kw.pop("look", 1.3),
                  kw.pop("xfade", 0.1), 0.0, **kw)
    out = []

    def drain():
        while True:
            b = st.outq.get()
            if b is None:
                break
            out.append(b)

    t = threading.Thread(target=st.worker, daemon=True)
    d = threading.Thread(target=drain, daemon=True)
    t.start()
    d.start()
    pad = st.stride + st.look + st.xfade
    padded = np.concatenate(
        [sig, np.zeros((sep.ch, pad), dtype=np.float32)], axis=1
    )
    for i in range(0, padded.shape[1], chunk):
        st.feed(padded[:, i : i + chunk])
    st.finish_input()
    t.join()
    d.join()
    return st, (np.concatenate([st.mixdown(b) for b in out], axis=1)
                if out else np.zeros((sep.ch, 0), dtype=np.float32))


def test_identity():
    """Stream must reconstruct its input exactly, at any feed granularity."""
    sep = fake_sep()
    n = 44100 * 40
    sig = (np.sin(2 * np.pi * np.arange(n) / 44100 * 0.5)
           * np.linspace(0, 1, n)).astype(np.float32)[None, :]
    for tag, mult in (("1 stride", 1.0), ("odd fraction", 0.37), ("tiny", 0.05)):
        st, y = run_stream(sep, sig, max(1, int(44100 * 4.0 * mult)))
        lag = st.look + st.xfade
        ya = y[0, lag:]
        m = min(sig.shape[1], len(ya))
        r, e = sig[0, :m], ya[:m]
        snr = 10 * np.log10((r ** 2).mean() / (((r - e) ** 2).mean() + 1e-20))
        check(f"identity reconstruction, feed={tag}", snr > 100, f"{snr:.1f} dB")


def test_backlog():
    """A worker slower than realtime must not accumulate unbounded lag."""
    SR, WIN, STRIDE, FEED, PROC = 1000, 1000, 100, 0.10, 0.15
    for max_lag, want_bounded in ((None, False), (0.3, True)):
        f = types.SimpleNamespace(sr=SR, ch=1, win=WIN)
        f.separate = lambda w: (time.sleep(PROC), np.stack([w] * 4))[1]
        st = Stream(f, dict(G.GAINS), STRIDE / SR, 0.0, 0.0, 0.0,
                      max_lag_s=max_lag)
        # Bind the queue explicitly: a lambda closing over `st` would late-bind
        # to whichever Stream the loop is on, leaving one queue undrained.
        def drain(q=st.outq):
            while q.get() is not None:
                pass

        threading.Thread(target=st.worker, daemon=True).start()
        threading.Thread(target=drain, daemon=True).start()
        for _ in range(30):
            st.feed(np.zeros((1, STRIDE), dtype=np.float32))
            time.sleep(FEED)
        lag = ((st.base + st.inbuf.shape[1]) - st.next_end) / SR
        st.finish_input()
        if want_bounded:
            check("backlog bounded when slow", lag <= 0.5, f"lag {lag:.2f}s")
        else:
            check("backlog grows unbounded without max_lag", lag > 0.5,
                  f"lag {lag:.2f}s")


def test_detents():
    """ADC noise must never move a detent, and a sweep must reach them all."""
    rng = np.random.default_rng(0)
    for tag, centre in (("mid-travel", 0.50), ("on a boundary", 0.525)):
        held, changes = None, 0
        for _ in range(2000):
            new = G.detent(centre + rng.normal(0, 0.002), held)
            changes += held is not None and new != held
            held = new
        check(f"detent stable, parked {tag}", changes == 0, f"{changes} changes")
    held, seen = None, []
    for pos in np.linspace(0, 1, 4001):
        held = G.detent(pos, held)
        if not seen or seen[-1] != held:
            seen.append(held)
    check("sweep visits every detent", len(seen) == G.POT_STEPS + 1,
          f"{len(seen)} of {G.POT_STEPS + 1}")


def test_gain_ladder():
    g = sorted({round(G.pot_gain(d / G.POT_STEPS), 6)
                for d in range(G.POT_STEPS + 1)})
    check("gain ladder distinct", len(g) == G.POT_STEPS + 1, f"{len(g)} values")
    check("ladder spans mute..max", g[0] == 0.0 and abs(g[-1] - G.GAIN_MAX) < 1e-6)


def test_entrypoint():
    """The CLI must actually be runnable. The refactor once dropped the
    __main__ guard, so the module imported fine and silently did nothing."""
    src = (HERE / "separate_stream.py").read_text(encoding="utf-8")
    check("entry point present", '__name__ == "__main__"' in src
          and "main()" in src.split('__name__ == "__main__"')[-1])


def test_offline(model, audio, repo):
    """End-to-end with the real model: stems must sum back to the mix."""
    sep = Separator(model, repo, threads=8)
    wav = audio_io.read_audio(audio, sep.sr, sep.ch)
    st, y = run_stream(sep, wav, int(sep.sr * 4.0))
    lag = st.look + st.xfade
    ya = y[:, lag:]
    m = min(wav.shape[1], ya.shape[1])
    r, e = wav[:, :m], ya[:, :m]
    snr = 10 * np.log10((r ** 2).mean() / (((r - e) ** 2).mean() + 1e-20))
    check("offline sum-of-stems vs mix", snr > 8.0, f"{snr:.2f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--audio", default="")
    ap.add_argument("--repo", default="..")
    args = ap.parse_args()

    print("structural checks (no model, no audio device)")
    test_identity()
    test_backlog()
    test_detents()
    test_gain_ladder()
    test_entrypoint()

    if args.model and args.audio:
        print("end-to-end check")
        test_offline(args.model, args.audio, args.repo)

    print()
    if FAILED:
        print(f"FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
