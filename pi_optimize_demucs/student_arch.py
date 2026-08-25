"""Size candidate student architectures BEFORE committing to distillation.

The point: training is the expensive step, so first prove some student is
actually fast enough on the Pi. If no student in a sane quality range hits the
target, distillation is not worth starting.

Budget maths from the measured Pi run: the teacher does 176.4 GMAC/window at
core RTF 2.01x (fp32 simplified). If time scaled purely with MACs, 0.8x needs

    176.4 * (0.8 / 2.01) = 70.2 GMAC/window     -> a 2.5x MAC cut

The Pi is bandwidth-bound though, and narrowing channels cuts activation bytes
LINEARLY while cutting conv MACs QUADRATICALLY -- so a channel-width cut should
beat the pure-MAC projection. That is the main reason width is the right knob.

Run: python student_arch.py
"""

import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th
import torch.nn as nn

from demucs.htdemucs import HTDemucs
from onnx_export import Core, front, get_htdemucs

TEACHER_GMAC = 176.4  # measured, one 7.8s window
TEACHER_RTF = 2.01  # measured on Pi, fp32 simplified, 1 thread
TARGET_RTF = 0.8


def count(core, ctx):
    macs = defaultdict(int)
    handles = []
    for name, mod in core.named_modules():
        if isinstance(
            mod, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
        ):
            k = 1
            for s in mod.kernel_size:
                k *= s

            def mkc(mod=mod, k=k):
                def h(m, i, o):
                    o = o[0] if isinstance(o, tuple) else o
                    macs["conv"] += o.numel() * (m.in_channels // m.groups) * k

                return h

            handles.append(mod.register_forward_hook(mkc()))
        elif isinstance(mod, nn.Linear):

            def mkl(mod=mod):
                def h(m, i, o):
                    o = o[0] if isinstance(o, tuple) else o
                    macs["linear"] += o.numel() * m.in_features

                return h

            handles.append(mod.register_forward_hook(mkl()))
        elif isinstance(mod, nn.MultiheadAttention):

            def mka(mod=mod):
                def h(m, i, o):
                    q = i[0]
                    L = q.shape[1] if getattr(m, "batch_first", False) else q.shape[0]
                    macs["attn"] += 2 * L * L * q.shape[-1]

                return h

            handles.append(mod.register_forward_hook(mka()))
    with th.no_grad():
        core(ctx["x"], ctx["xt"])
    for h in handles:
        h.remove()
    return macs


def teacher_config(m):
    """Read the knobs that actually differ from HTDemucs defaults."""
    keys = [
        "channels",
        "depth",
        "growth",
        "nfft",
        "bottom_channels",
        "t_layers",
        "t_heads",
        "dconv_depth",
    ]
    cfg = {}
    for k in keys:
        cfg[k] = getattr(m, k, None)
    # some are only on submodules
    ct = m.crosstransformer
    cfg["t_layers"] = len(getattr(ct, "layers", []) or [])
    return cfg


def build(seconds, **kw):
    m = HTDemucs(
        sources=["drums", "bass", "other", "vocals"],
        segment=Fraction(seconds).limit_denominator(1000),
        **kw,
    )
    m.eval()
    return m


def evaluate(tag, m, seconds):
    core = Core(m).eval()
    ctx = front(m, th.randn(1, m.audio_channels, int(m.segment * m.samplerate)))
    macs = count(core, ctx)
    tot = sum(macs.values())
    params = sum(p.numel() for p in m.parameters())
    # activation proxy: the biggest spec tensor, which drives bandwidth
    act_mb = ctx["x"].numel() * 4 / 1e6
    proj = TEACHER_RTF * tot / TEACHER_GMAC / 1e9 * 1e9 / 1e0
    proj = TEACHER_RTF * (tot / 1e9) / TEACHER_GMAC
    return dict(
        tag=tag,
        params=params / 1e6,
        gmac=tot / 1e9,
        conv=macs["conv"] / 1e9,
        lin=macs["linear"] / 1e9,
        attn=macs["attn"] / 1e9,
        act_mb=act_mb,
        proj_rtf=proj,
    )


def main():
    teacher = get_htdemucs()
    seg = float(teacher.segment)
    cfg = teacher_config(teacher)
    print("teacher (pretrained htdemucs) config:")
    for k, v in cfg.items():
        print(f"   {k:>16} = {v}")

    rows = [evaluate("teacher(48ch,512bc)", teacher, seg)]

    # Candidates. channels drives conv cost (~quadratic); bottom_channels and
    # t_layers drive the transformer; both also shrink activations.
    cands = [
        ("ch40 bc384 t5", dict(channels=40, bottom_channels=384, t_layers=5)),
        ("ch32 bc384 t5", dict(channels=32, bottom_channels=384, t_layers=5)),
        ("ch32 bc256 t4", dict(channels=32, bottom_channels=256, t_layers=4)),
        ("ch24 bc256 t4", dict(channels=24, bottom_channels=256, t_layers=4)),
        ("ch24 bc192 t3", dict(channels=24, bottom_channels=192, t_layers=3)),
        ("ch16 bc192 t3", dict(channels=16, bottom_channels=192, t_layers=3)),
    ]
    for tag, kw in cands:
        try:
            rows.append(evaluate(tag, build(seg, t_heads=8, **kw), seg))
        except Exception as e:
            print(f"  {tag} FAILED: {type(e).__name__}: {e}")

    hdr = (
        f"{'variant':<20}{'params M':>10}{'GMAC':>8}{'conv':>8}{'lin':>7}"
        f"{'attn':>7}{'specMB':>8}{'cut':>7}{'proj RTF':>10}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    base = rows[0]
    for r in rows:
        mark = "  <= TARGET" if r["proj_rtf"] <= TARGET_RTF else ""
        print(
            f"{r['tag']:<20}{r['params']:>10.1f}{r['gmac']:>8.1f}"
            f"{r['conv']:>8.1f}{r['lin']:>7.1f}{r['attn']:>7.1f}"
            f"{r['act_mb']:>8.1f}{base['gmac']/r['gmac']:>6.2f}x"
            f"{r['proj_rtf']:>10.2f}{mark}"
        )

    print(
        f"\nproj RTF = measured teacher {TEACHER_RTF}x scaled by MAC ratio "
        f"(conservative: ignores the bandwidth win from narrower activations)."
    )


if __name__ == "__main__":
    main()
