"""Structured channel pruning: initialise a narrow student from the htdemucs teacher.

Why not random init: distillation from scratch on MUSDB (~10 h) is slow and
data-starved. Keeping the teacher's most important channels gives the student a
working feature hierarchy on day zero, so distillation becomes fine-tuning.

The three constraints that make this non-trivial
------------------------------------------------
1. **GLU pairing.** `rewrite` convs emit 2C channels which GLU splits as
   a * sigmoid(b). Keeping output index i means keeping BOTH row i and row C+i,
   or the gate no longer matches its value.
2. **Skip connections.** decoder[j] consumes encoder[depth-1-j]'s output as a
   skip and adds it. Both must therefore use the SAME channel index set.
3. **dconv bottlenecks.** Conv1d(C -> C/compress) then Conv1d(C/compress -> 2C).
   The hidden width is pruned independently, but its input/output indices must
   line up with the enclosing layer's channel set.

Importance = L1 norm of the teacher's conv weight over every dim except the
output-channel dim. Indices are kept sorted so ordering stays comparable to the
teacher (matters for the residual adds).

Run:
    python prune_init.py --channels 24 --bottom-channels 256 --t-layers 4 \
                         --out models/student_pruned.pt --verify
"""

import argparse
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import torch as th
import torch.nn as nn

from demucs.htdemucs import HTDemucs
from onnx_export import get_htdemucs


# ---------------------------------------------------------------- index picking
def topk_idx(weight, k, out_dim=0):
    """Indices of the k most important slices along `out_dim`, sorted ascending."""
    dims = [d for d in range(weight.dim()) if d != out_dim]
    imp = weight.abs().sum(dim=dims) if dims else weight.abs()
    idx = th.topk(imp, k).indices
    return th.sort(idx).values


def glu_pair(idx, c_teacher):
    """A 2C tensor split by GLU into [values | gates]: keep i and C+i together."""
    return th.cat([idx, idx + c_teacher])


class Plan:
    """Per-level channel index sets, computed once so every consumer agrees."""

    def __init__(self, teacher, student):
        self.t, self.s = teacher, student
        self.enc = {}  # branch -> {level: idx}
        self.bt = None  # transformer bottom_channels idx
        self.ffn = None  # transformer FFN hidden idx
        self._build()

    def _levels(self, branch):
        tmods = getattr(self.t, branch)
        smods = getattr(self.s, branch)
        out = {}
        for i, (tm, sm) in enumerate(zip(tmods, smods)):
            tw, sw = tm.conv.weight, sm.conv.weight
            # conv weight is [out, in, ...] for both Conv1d/Conv2d
            out[i] = topk_idx(tw, sw.shape[0], out_dim=0)
        return out

    def _build(self):
        self.enc["encoder"] = self._levels("encoder")
        self.enc["tencoder"] = self._levels("tencoder")

        tct, sct = self.t.crosstransformer, self.s.crosstransformer
        if self.t.bottom_channels:
            self.bt = topk_idx(
                self.t.channel_upsampler.weight,
                self.s.channel_upsampler.weight.shape[0],
                0,
            )
        # FFN hidden width
        tl, sl = tct.layers[0], sct.layers[0]
        if hasattr(tl, "linear1"):
            self.ffn = topk_idx(tl.linear1.weight, sl.linear1.weight.shape[0], 0)

    def enc_idx(self, branch, level):
        return self.enc[branch][level]

    def dec_chin(self, branch, j, depth):
        """decoder[j] input channels == encoder[depth-1-j] output channels."""
        enc_branch = "encoder" if branch == "decoder" else "tencoder"
        return self.enc_idx(enc_branch, depth - 1 - j)


# ------------------------------------------------------------------- transfer
class Transfer:
    def __init__(self, teacher, student, plan):
        self.t, self.s, self.p = teacher, student, plan
        self.copied = self.total = 0
        self.skipped = []
        self.touched = set()

    def _slice(self, tw, idx_out=None, idx_in=None, out_dim=0, in_dim=1):
        w = tw
        if idx_out is not None:
            w = w.index_select(out_dim, idx_out)
        if idx_in is not None:
            w = w.index_select(in_dim, idx_in)
        return w

    def put(self, sparam, value, name):
        self.total += sparam.numel()
        self.touched.add(id(sparam))
        if tuple(value.shape) != tuple(sparam.shape):
            self.skipped.append(
                f"{name}: got {tuple(value.shape)} " f"want {tuple(sparam.shape)}"
            )
            return
        with th.no_grad():
            sparam.copy_(value)
        self.copied += sparam.numel()

    def untouched(self):
        """Student params no transfer rule ever reached -- these stay random,
        which silently poisons the init. Must be empty (or intentionally so)."""
        out = []
        for name, p in self.s.named_parameters():
            if id(p) not in self.touched:
                out.append((name, tuple(p.shape), p.numel()))
        return out

    def conv(self, tm, sm, idx_out, idx_in, name, transposed=False, glu=False):
        """Transfer a conv (or conv_transpose) plus bias.

        ConvTranspose weight is [in, out, ...] -- the axes are swapped relative
        to Conv, which is an easy and silent source of corruption.
        """
        c_t = tm.weight.shape[1] if transposed else tm.weight.shape[0]
        o = idx_out
        if glu and o is not None:
            o = glu_pair(o, c_t // 2 if not transposed else c_t // 2)
        od, idm = (1, 0) if transposed else (0, 1)
        w = self._slice(tm.weight, o, idx_in, out_dim=od, in_dim=idm)
        self.put(sm.weight, w, name + ".weight")
        if tm.bias is not None and sm.bias is not None:
            b = tm.bias if o is None else tm.bias.index_select(0, o)
            self.put(sm.bias, b, name + ".bias")

    def norm(self, tm, sm, idx, name, glu=False):
        if tm is None or sm is None or not hasattr(tm, "weight") or tm.weight is None:
            return
        i = idx
        if glu and i is not None:
            i = glu_pair(i, tm.weight.shape[0] // 2)
        for attr in ("weight", "bias"):
            tw, sw = getattr(tm, attr, None), getattr(sm, attr, None)
            if tw is None or sw is None:
                continue
            v = tw if i is None else tw.index_select(0, i)
            self.put(sw, v, f"{name}.{attr}")

    def dconv(self, td, sd, idx_c, name):
        """DConv layer d: Conv1d(C->hidden), norm(hidden), act,
        Conv1d(hidden->2C), norm(2C), GLU, LayerScale(C)."""
        if td is None or sd is None:
            return
        for d, (tl, sl) in enumerate(zip(td.layers, sd.layers)):
            t0, s0 = tl[0], sl[0]  # Conv1d C -> hidden
            hid = topk_idx(t0.weight, s0.weight.shape[0], 0)
            self.conv(t0, s0, hid, idx_c, f"{name}.layers.{d}.0")
            self.norm(tl[1], sl[1], hid, f"{name}.layers.{d}.1")
            t3, s3 = tl[3], sl[3]  # Conv1d hidden -> 2C
            self.conv(t3, s3, idx_c, hid, f"{name}.layers.{d}.3", glu=True)
            self.norm(tl[4], sl[4], idx_c, f"{name}.layers.{d}.4", glu=True)
            # LayerScale holds a per-channel `scale` vector
            for tm2, sm2 in zip(tl.modules(), sl.modules()):
                if hasattr(tm2, "scale") and hasattr(sm2, "scale"):
                    self.put(
                        sm2.scale,
                        tm2.scale.index_select(0, idx_c),
                        f"{name}.layers.{d}.scale",
                    )
                    break

    # ---------------------------------------------------------------- branches
    def encoder_branch(self, branch):
        tmods, smods = getattr(self.t, branch), getattr(self.s, branch)
        prev = None  # input channels are raw audio/spec: keep all
        for i, (tm, sm) in enumerate(zip(tmods, smods)):
            idx = self.p.enc_idx(branch, i)
            nm = f"{branch}.{i}"
            self.conv(tm.conv, sm.conv, idx, prev, nm + ".conv")
            self.norm(
                getattr(tm, "norm1", None),
                getattr(sm, "norm1", None),
                idx,
                nm + ".norm1",
            )
            if getattr(tm, "rewrite", None) is not None:
                self.conv(tm.rewrite, sm.rewrite, idx, idx, nm + ".rewrite", glu=True)
                self.norm(
                    getattr(tm, "norm2", None),
                    getattr(sm, "norm2", None),
                    idx,
                    nm + ".norm2",
                    glu=True,
                )
            self.dconv(
                getattr(tm, "dconv", None),
                getattr(sm, "dconv", None),
                idx,
                nm + ".dconv",
            )
            prev = idx

    def decoder_branch(self, branch, depth):
        tmods, smods = getattr(self.t, branch), getattr(self.s, branch)
        n = len(tmods)
        for j, (tm, sm) in enumerate(zip(tmods, smods)):
            chin = self.p.dec_chin(branch, j + (depth - n), depth)
            nm = f"{branch}.{j}"
            if getattr(tm, "rewrite", None) is not None:
                self.conv(tm.rewrite, sm.rewrite, chin, chin, nm + ".rewrite", glu=True)
                self.norm(
                    getattr(tm, "norm1", None),
                    getattr(sm, "norm1", None),
                    chin,
                    nm + ".norm1",
                    glu=True,
                )
            self.dconv(
                getattr(tm, "dconv", None),
                getattr(sm, "dconv", None),
                chin,
                nm + ".dconv",
            )
            # conv_tr: last layer emits the source channels -> keep all of them
            is_last = j == n - 1
            chout = (
                None if is_last else self.p.dec_chin(branch, j + 1 + (depth - n), depth)
            )
            self.conv(
                tm.conv_tr, sm.conv_tr, chout, chin, nm + ".conv_tr", transposed=True
            )
            if not is_last:
                self.norm(
                    getattr(tm, "norm2", None),
                    getattr(sm, "norm2", None),
                    chout,
                    nm + ".norm2",
                )

    def transformer(self):
        t, s, bt, ffn = self.t, self.s, self.p.bt, self.p.ffn
        if t.crosstransformer is None:
            return
        if t.bottom_channels:
            self.conv(
                t.channel_upsampler,
                s.channel_upsampler,
                bt,
                self.p.enc_idx("encoder", 3),
                "channel_upsampler",
            )
            self.conv(
                t.channel_upsampler_t,
                s.channel_upsampler_t,
                bt,
                self.p.enc_idx("tencoder", 3),
                "channel_upsampler_t",
            )
            self.conv(
                t.channel_downsampler,
                s.channel_downsampler,
                self.p.enc_idx("encoder", 3),
                bt,
                "channel_downsampler",
            )
            self.conv(
                t.channel_downsampler_t,
                s.channel_downsampler_t,
                self.p.enc_idx("tencoder", 3),
                bt,
                "channel_downsampler_t",
            )

        for nm in ("norm_in", "norm_in_t", "norm_out", "norm_out_t"):
            self.norm(
                getattr(t.crosstransformer, nm, None),
                getattr(s.crosstransformer, nm, None),
                bt,
                f"crosstransformer.{nm}",
            )

        for attr in ("layers", "layers_t"):
            tls, sls = getattr(t.crosstransformer, attr), getattr(
                s.crosstransformer, attr
            )
            # The stack ALTERNATES layer types: even index = self-attention,
            # odd index = cross-attention. An evenly-spaced subset breaks that
            # parity and maps a self-attn student layer onto a cross-attn
            # teacher layer, so the tensors silently never match. Taking the
            # first N layers preserves the alternation.
            for si in range(len(sls)):
                self.tlayer(tls[si], sls[si], bt, ffn, f"{attr}.{si}")

    def tlayer(self, tl, sl, bt, ffn, name):
        # attention: in_proj_weight is [3*E, E] stacked as q|k|v
        for an in ("self_attn", "cross_attn", "attn"):
            ta, sa = getattr(tl, an, None), getattr(sl, an, None)
            if ta is None or sa is None:
                continue
            E = ta.embed_dim
            if getattr(ta, "in_proj_weight", None) is not None:
                rows = th.cat([bt, bt + E, bt + 2 * E])
                w = ta.in_proj_weight.index_select(0, rows).index_select(1, bt)
                self.put(sa.in_proj_weight, w, f"{name}.{an}.in_proj_weight")
                if ta.in_proj_bias is not None:
                    self.put(
                        sa.in_proj_bias,
                        ta.in_proj_bias.index_select(0, rows),
                        f"{name}.{an}.in_proj_bias",
                    )
            self.put(
                sa.out_proj.weight,
                ta.out_proj.weight.index_select(0, bt).index_select(1, bt),
                f"{name}.{an}.out_proj.weight",
            )
            if ta.out_proj.bias is not None:
                self.put(
                    sa.out_proj.bias,
                    ta.out_proj.bias.index_select(0, bt),
                    f"{name}.{an}.out_proj.bias",
                )

        if hasattr(tl, "linear1"):
            self.conv_like_linear(tl.linear1, sl.linear1, ffn, bt, f"{name}.linear1")
            self.conv_like_linear(tl.linear2, sl.linear2, bt, ffn, f"{name}.linear2")
        for nn_ in ("norm1", "norm2", "norm3", "norm_out"):
            self.norm(
                getattr(tl, nn_, None), getattr(sl, nn_, None), bt, f"{name}.{nn_}"
            )
        for sc in ("gamma_1", "gamma_2"):
            tg, sg = getattr(tl, sc, None), getattr(sl, sc, None)
            if tg is not None and sg is not None and hasattr(tg, "scale"):
                self.put(sg.scale, tg.scale.index_select(0, bt), f"{name}.{sc}")

    def conv_like_linear(self, tm, sm, idx_out, idx_in, name):
        w = tm.weight.index_select(0, idx_out).index_select(1, idx_in)
        self.put(sm.weight, w, name + ".weight")
        if tm.bias is not None:
            self.put(sm.bias, tm.bias.index_select(0, idx_out), name + ".bias")

    def run(self, depth):
        self.encoder_branch("encoder")
        self.encoder_branch("tencoder")
        self.decoder_branch("decoder", depth)
        self.decoder_branch("tdecoder", depth)
        self.transformer()
        # freq embedding lives at encoder level 0
        if getattr(self.t, "freq_emb", None) is not None:
            ti = self.t.freq_emb.embedding.weight
            si = self.s.freq_emb.embedding.weight
            self.put(si, ti.index_select(1, self.p.enc_idx("encoder", 0)), "freq_emb")


def build_student(teacher, channels, bottom_channels, t_layers, t_heads):
    return HTDemucs(
        sources=list(teacher.sources),
        audio_channels=teacher.audio_channels,
        channels=channels,
        bottom_channels=bottom_channels,
        t_layers=t_layers,
        t_heads=t_heads,
        nfft=teacher.nfft,
        depth=teacher.depth,
        segment=teacher.segment,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=24)
    ap.add_argument("--bottom-channels", type=int, default=256)
    ap.add_argument("--t-layers", type=int, default=4)
    ap.add_argument("--t-heads", type=int, default=8)
    ap.add_argument("--out", default="pi_optimize_demucs/models/student_pruned.pt")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="compare pruned-init vs random-init output against the "
        "teacher on real audio -- the acceptance test",
    )
    ap.add_argument("--audio", default="profile/calibration_mp3")
    args = ap.parse_args()

    teacher = get_htdemucs().eval()
    student = build_student(
        teacher, args.channels, args.bottom_channels, args.t_layers, args.t_heads
    ).eval()
    tp = sum(p.numel() for p in teacher.parameters()) / 1e6
    sp = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"teacher {tp:.1f}M -> student {sp:.1f}M params " f"({tp/sp:.2f}x smaller)")

    plan = Plan(teacher, student)
    xfer = Transfer(teacher, student, plan)
    xfer.run(teacher.depth)

    pct = 100 * xfer.copied / xfer.total if xfer.total else 0
    print(
        f"transferred {xfer.copied/1e6:.2f}M / {xfer.total/1e6:.2f}M "
        f"student params ({pct:.1f}%)"
    )
    if xfer.skipped:
        print(f"  {len(xfer.skipped)} tensor(s) shape-mismatched:")
        for s in xfer.skipped[:12]:
            print(f"    {s}")
    un = xfer.untouched()
    if un:
        tot = sum(n for _, _, n in un)
        print(
            f"  {len(un)} tensor(s) NEVER REACHED by a rule "
            f"({tot/1e6:.2f}M params, left random):"
        )
        for nm, shp, _ in un[:20]:
            print(f"    {nm} {shp}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    th.save(
        {
            "state_dict": student.state_dict(),
            "arch": dict(
                channels=args.channels,
                bottom_channels=args.bottom_channels,
                t_layers=args.t_layers,
                t_heads=args.t_heads,
            ),
        },
        out,
    )
    print(f"wrote {out}")

    if args.verify:
        verify(teacher, student, args)


def exactness_check(teacher, student, plan, sr_input):
    """Hard correctness test for the slicing machinery.

    encoder[0].conv sees an UNPRUNED input, so if the transfer is right its
    output on kept channel k must equal the teacher's output on channel
    idx[k] *exactly* (up to float noise). Any index/axis mistake shows up here
    immediately, unlike an end-to-end SNR which mixes in the real (and expected)
    loss from removing capacity.
    """
    idx = plan.enc_idx("encoder", 0)
    with th.no_grad():
        tw = teacher.encoder[0].conv(sr_input)
        sw = student.encoder[0].conv(sr_input)
    ref = tw.index_select(1, idx)
    err = (ref - sw).abs().max().item()
    ok = err < 1e-4
    print(f"\nexactness: encoder[0].conv student vs teacher[kept channels]")
    print(f"  max abs diff = {err:.2e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def verify(teacher, student, args):
    """Acceptance test: pruned init must beat random init by a wide margin."""
    from demucs.audio import AudioFile

    rnd = build_student(
        teacher, args.channels, args.bottom_channels, args.t_layers, args.t_heads
    ).eval()

    sr = teacher.samplerate
    seg = int(teacher.segment * sr)
    mp3 = sorted(Path(args.audio).glob("*.mp3"))[0]
    wav = AudioFile(mp3).read(
        streams=0, samplerate=sr, channels=teacher.audio_channels
    )[:, :seg][None]

    def snr(ref, x):
        return float(10 * th.log10(th.mean(ref**2) / (th.mean((ref - x) ** 2) + 1e-20)))

    # spec-branch input the encoder actually consumes
    from onnx_export import front

    ctx = front(teacher, wav)
    exactness_check(teacher, student, Plan(teacher, student), ctx["x"])

    with th.no_grad():
        ref = teacher(wav)
        a = student(wav)
        b = rnd(wav)
    print(f"\nverify on {mp3.name} (one {float(teacher.segment):.1f}s window)")
    print(f"  pruned-init vs teacher : {snr(ref, a):7.2f} dB")
    print(f"  random-init vs teacher : {snr(ref, b):7.2f} dB")
    print("  (pruned should be clearly higher; both will be poor until distilled)")


if __name__ == "__main__":
    main()
