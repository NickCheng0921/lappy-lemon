"""Export the htdemucs *core* (encoder/transformer/decoder) to ONNX and verify.

Why only the core: htdemucs' forward starts with an STFT (complex tensor) and
ends with complex masking + iSTFT. ONNX has no complex dtype, so the whole model
can't be exported directly. But all of the *compute* (the conv encoder, the
cross-transformer, the conv decoder -- ~all 130 GMAC) is plain real-valued math.
So we export just that middle, and keep the cheap STFT/iSTFT/normalization in
torch. The core takes two normalized real tensors (freq branch `x`, time branch
`xt`) and returns two real tensors, exactly matching htdemucs.forward lines
561-623 in vendor/demucs/htdemucs.py.

Usage (run in your WSL env that already has torch + demucs working):
    pip install onnx onnxruntime            # onnxruntime-tools optional
    python profile/onnx_export.py --export   -o profile/htdemucs_core.onnx
    python profile/onnx_export.py --verify   -o profile/htdemucs_core.onnx

--export writes the .onnx (do this on the fast box, e.g. the 7600x).
--verify runs ORT vs torch on a random segment and prints max abs error.
Then scp the .onnx to the Pi and run ORT there.
"""

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))

import torch as th
import torch.nn as nn
from einops import rearrange

from demucs.pretrained import get_model
from demucs.apply import BagOfModels


def get_htdemucs(name="htdemucs"):
    model = get_model(name)
    m = model.models[0] if isinstance(model, BagOfModels) else model
    m.eval()
    return m


class Core(nn.Module):
    """The real-valued middle of htdemucs.forward.

    forward(x, xt): x is the normalized magnitude (freq branch), xt is the
    normalized time-domain mix. Returns (x, xt) raw core outputs, before the
    view/denormalize/mask/ispec steps that stay in torch outside ONNX.

    This mirrors vendor/demucs/htdemucs.py forward() lines 561-623 exactly.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, xt):
        m = self.m
        saved = []
        saved_t = []
        lengths = []
        lengths_t = []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = th.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)

        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_upsampler_t(xt)

            x, xt = m.crosstransformer(x, xt)

            if m.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_downsampler_t(xt)

        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        return x, xt


def front(m, mix):
    """Torch front end for one segment: pad to training length, STFT, magnitude,
    normalize both branches. Returns the two normalized real Core inputs plus the
    ctx tensors needed to reconstruct. Mirrors htdemucs.forward lines 528-554."""
    sr = m.samplerate
    training_length = int(m.segment * sr)
    length_pre_pad = None
    if mix.shape[-1] < training_length:
        length_pre_pad = mix.shape[-1]
        mix = th.nn.functional.pad(mix, (0, training_length - length_pre_pad))

    z = m._spec(mix)
    mag = m._magnitude(z).to(mix.device)
    x = mag
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / (1e-5 + std)

    xt = mix
    meant = xt.mean(dim=(1, 2), keepdim=True)
    stdt = xt.std(dim=(1, 2), keepdim=True)
    xt = (xt - meant) / (1e-5 + stdt)
    return dict(
        x=x, xt=xt, z=z, mix=mix, mean=mean, std=std, meant=meant, stdt=stdt,
        training_length=training_length, length_pre_pad=length_pre_pad,
    )


def make_core_inputs(m, seconds=None, device="cpu"):
    """Build Core inputs on a random segment (for --verify / --export shapes)."""
    sr = m.samplerate
    ch = m.audio_channels
    training_length = int(m.segment * sr)
    n = training_length if seconds is None else int(seconds * sr)
    mix = th.randn(1, ch, n, device=device)
    return front(m, mix)


def reconstruct(m, core_x, core_xt, ctx):
    """Torch tail: denormalize, mask, iSTFT, add time branch. Mirrors
    htdemucs.forward lines 624-660 (non-mps, use_train_segment eval path)."""
    B = ctx["x"].shape[0]
    Fq = ctx["x"].shape[2]
    T = ctx["x"].shape[3]
    S = len(m.sources)
    x = core_x.view(B, S, -1, Fq, T)
    x = x * ctx["std"][:, None] + ctx["mean"][:, None]
    zout = m._mask(ctx["z"], x)
    x = m._ispec(zout, ctx["training_length"])

    xt = core_xt.view(B, S, -1, ctx["training_length"])
    xt = xt * ctx["stdt"][:, None] + ctx["meant"][:, None]
    out = xt + x
    if ctx["length_pre_pad"]:
        out = out[..., : ctx["length_pre_pad"]]
    return out


class OrtCoreModel(nn.Module):
    """A drop-in for htdemucs that runs the heavy core in ONNX Runtime while
    keeping the STFT/iSTFT tail in torch. Exposes the attributes apply_model
    needs (sources, samplerate, audio_channels, segment) so demucs' own
    windowing/overlap pipeline can drive it unchanged."""

    def __init__(self, m, sess):
        super().__init__()
        self.m = m
        self.sess = sess
        self.sources = m.sources
        self.samplerate = m.samplerate
        self.audio_channels = m.audio_channels
        self.segment = m.segment

    def forward(self, mix):
        ctx = front(self.m, mix)
        outs = self.sess.run(
            ["spec_out", "time_out"],
            {"mag": ctx["x"].numpy(), "mix_t": ctx["xt"].numpy()},
        )
        core_x = th.from_numpy(outs[0])
        core_xt = th.from_numpy(outs[1])
        return reconstruct(self.m, core_x, core_xt, ctx)


def _patch_unflatten():
    """torch 2.0.1's ONNX exporter has no symbolic for aten::unflatten (used in
    the decomposed MHA path). Replace it with an equivalent reshape, which the
    exporter supports. Traced with static shapes so it's exact. Returns the
    original for restoration."""
    orig = th.Tensor.unflatten

    def _unflatten(self, dim, sizes):
        if dim < 0:
            dim += self.dim()
        shape = list(self.shape)
        new_shape = shape[:dim] + list(sizes) + shape[dim + 1 :]
        return self.reshape(new_shape)

    th.Tensor.unflatten = _unflatten
    return orig


def _patch_sdpa():
    """torch 2.0.1's exporter can't lower the fused
    aten::scaled_dot_product_attention. Replace it with the plain-math
    equivalent (matmul + softmax + matmul), which exports cleanly. eval() means
    dropout_p is 0. Returns the original for restoration."""
    import torch.nn.functional as F

    orig = F.scaled_dot_product_attention

    def _sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
        attn = (query @ key.transpose(-2, -1)) * scale_factor
        if attn_mask is not None:
            if attn_mask.dtype == th.bool:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            else:
                attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        return attn @ value

    F.scaled_dot_product_attention = _sdpa
    return orig


def cmd_export(args):
    m = get_htdemucs(args.name)
    core = Core(m).eval()
    ctx = make_core_inputs(m)
    orig_unflatten = _patch_unflatten()
    orig_sdpa = _patch_sdpa()
    # NOTE: do NOT wrap in th.no_grad(). nn.MultiheadAttention only falls back
    # to the ONNX-exportable *decomposed* attention when grad is enabled and
    # params require grad; under no_grad it uses the fused
    # aten::_native_multi_head_attention kernel, which torch 2.0.1's exporter
    # cannot lower. eval() (dropout off) + grad enabled is what we want here.
    with th.enable_grad():
        th.onnx.export(
            core,
            (ctx["x"], ctx["xt"]),
            args.out,
            input_names=["mag", "mix_t"],
            output_names=["spec_out", "time_out"],
            opset_version=args.opset,
            do_constant_folding=True,
        )
    th.Tensor.unflatten = orig_unflatten
    import torch.nn.functional as F

    F.scaled_dot_product_attention = orig_sdpa
    print(f"wrote {args.out}")
    print("input shapes:", tuple(ctx["x"].shape), tuple(ctx["xt"].shape))


def _real_audio_mix(m, path):
    """One training-length window from a real mp3 -> [1, C, seg]. int8 models are
    calibrated on real-audio activation ranges, so verify on real audio, NOT the
    random noise that make_core_inputs uses (noise blows past calibrated ranges
    and makes a good int8 model look broken)."""
    from demucs.audio import AudioFile

    sr = m.samplerate
    seg = int(m.segment * sr)
    wav = AudioFile(path).read(streams=0, samplerate=sr, channels=m.audio_channels)
    start = min(seg, max(0, wav.shape[-1] - seg))  # skip the silent intro if we can
    return wav[:, start:start + seg][None]


def cmd_verify(args):
    import numpy as np
    import onnxruntime as ort

    m = get_htdemucs(args.name)
    if args.audio:
        mix = _real_audio_mix(m, args.audio)
        ctx = front(m, mix)
        print(f"verify input: real audio {args.audio}")
    else:
        ctx = make_core_inputs(m)
        print("verify input: random noise (use --audio <mp3> for a real-audio "
              "check; int8 models NEED this)")

    # torch reference: full forward on the same mix.
    with th.no_grad():
        ref = m(ctx["mix"])

    # ORT core + torch tail.
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])
    outs = sess.run(
        ["spec_out", "time_out"],
        {"mag": ctx["x"].numpy(), "mix_t": ctx["xt"].numpy()},
    )
    core_x = th.from_numpy(outs[0])
    core_xt = th.from_numpy(outs[1])
    with th.no_grad():
        got = reconstruct(m, core_x, core_xt, ctx)

    err = (got - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-9)
    # RMS-relative error tracks perceived quality far better than a single max
    # outlier -- this is the number to watch for int8.
    rms_rel = (((got - ref) ** 2).mean().sqrt()
               / ((ref ** 2).mean().sqrt() + 1e-9)).item()
    print(f"max abs err vs torch: {err:.3e}  (rel {rel:.3e})")
    print(f"RMS-relative err:     {rms_rel:.3e}")
    print("PASS" if err < 1e-3 else "CHECK: error larger than expected "
          "(expected for int8 -- judge by RMS-rel + a listen, not this)")


def peak_rss_mb():
    """Process-wide peak resident memory (high-water mark). Linux ru_maxrss is
    in KB, macOS in bytes. None on Windows (no `resource` module)."""
    try:
        import resource
    except ImportError:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def _fmt_mb(mb):
    return "n/a (Windows)" if mb is None else f"{mb:.0f} MB"


def _keep_op_types(exclude):
    """Dynamic-quantizable op types minus the excluded ones (e.g. exclude
    'MatMul' to keep the attention/transformer in fp32 and quantize only Conv)."""
    all_types = ["Conv", "MatMul", "Gemm"]
    ex = {s.strip() for s in exclude.split(",") if s.strip()}
    return [t for t in all_types if t not in ex]


def _preprocess(src, dst):
    """Shape inference + graph opt before quantizing. We exported with fully
    static shapes, so skip *symbolic* shape inference (it chokes on the
    ConvTranspose/Pad size relations and buys us nothing). Falls back to the raw
    model if pre-processing fails -- it's optional. Returns the path to quantize."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    try:
        print(f"pre-processing {src} ...")
        quant_pre_process(src, dst, skip_symbolic_shape=True)
        return dst
    except Exception as e:
        print(f"  pre-process skipped ({type(e).__name__}: {e}); using raw model")
        return src


def _iter_core_feeds(m, mp3_dir, max_windows):
    """Yield {'mag', 'mix_t'} numpy feeds from real audio: load every mp3 under
    mp3_dir, chop into non-overlapping training-length windows, run the torch
    front end. These are the actual activation distributions ORT calibrates on."""
    from demucs.audio import AudioFile
    from tqdm import tqdm

    sr = m.samplerate
    ch = m.audio_channels
    seg = int(m.segment * sr)
    mp3s = sorted(Path(mp3_dir).glob("*.mp3"))
    if not mp3s:
        raise FileNotFoundError(f"no .mp3 files in {mp3_dir}")
    print(f"calibration: {len(mp3s)} file(s) in {mp3_dir}")
    n = 0
    bar = tqdm(total=max_windows, desc="calibration windows", unit="win")
    for path in mp3s:
        # streams=0 -> [C, T]; the default slice(None) returns [S, C, T].
        wav = AudioFile(path).read(streams=0, samplerate=sr, channels=ch)  # [C, T]
        for start in range(0, wav.shape[-1] - seg + 1, seg):
            if n >= max_windows:
                bar.close()
                return
            window = wav[:, start:start + seg][None]  # [1, C, seg]
            ctx = front(m, window)
            yield {"mag": ctx["x"].numpy(), "mix_t": ctx["xt"].numpy()}
            n += 1
            bar.update(1)
    bar.close()


def _staged_static(model_in, model_out, reader, method, args):
    """Run the static-PTQ stages manually with peak-RSS prints between each, to
    find which stage OOMs: graph augmentation, the calibration forward, range
    compute, or the final quantize/save."""
    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization.calibrate import create_calibrator
    from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
    import onnx

    op_types = _keep_op_types(args.exclude_ops) if args.exclude_ops else None
    aug = args.quant_out + ".augmented.onnx"

    print(f"  [stage] start: peak RSS {_fmt_mb(peak_rss_mb())}")
    calibrator = create_calibrator(
        Path(model_in), op_types, augmented_model_path=aug,
        calibrate_method=method,
    )
    print(f"  [stage] after create_calibrator (augment): peak RSS "
          f"{_fmt_mb(peak_rss_mb())}")

    calibrator.collect_data(reader)
    print(f"  [stage] after collect_data (calib forward): peak RSS "
          f"{_fmt_mb(peak_rss_mb())}")

    tensors_range = calibrator.compute_data()
    print(f"  [stage] after compute_data (ranges): peak RSS "
          f"{_fmt_mb(peak_rss_mb())}")
    del calibrator

    model = onnx.load(model_in)
    quantizer = QDQQuantizer(
        model, args.per_channel, False, QuantType.QInt8, QuantType.QInt8,
        tensors_range, None, None, op_types, {},
    )
    quantizer.quantize_model()
    print(f"  [stage] after quantize_model: peak RSS {_fmt_mb(peak_rss_mb())}")
    quantizer.model.save_model_to_file(model_out, False)
    print(f"  [stage] after save: peak RSS {_fmt_mb(peak_rss_mb())}")
    Path(aug).unlink(missing_ok=True)


def cmd_quantize(args):
    """int8 quantization of the fp32 core. --static (default here) does PTQ with
    calibration over real audio in profile/calibration_mp3 -- the right tool for
    this conv-heavy model. --no-static falls back to dynamic quant (weights only,
    no calibration), which we found destroys quality on this CNN. Reads args.out
    and writes args.quant_out; then verify/bench/e2e with -o <quant_out>."""
    from onnxruntime.quantization import QuantType

    def _mb(p):
        return Path(p).stat().st_size / (1024 * 1024)

    src = args.out
    pre = args.quant_out + ".pre.onnx"
    to_quantize = _preprocess(src, pre)
    op_types = _keep_op_types(args.exclude_ops) if args.exclude_ops else None

    if args.static:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            quantize_static,
        )

        m = get_htdemucs(args.name)
        feeds = list(_iter_core_feeds(m, args.calib_dir, args.calib_windows))
        print(f"  collected {len(feeds)} calibration window(s)")

        class Reader(CalibrationDataReader):
            """Supports CalibStridedMinMax streaming: ORT calls set_range() to
            walk the data in chunks of `stride`, running collect_data (which
            merges ranges + frees activations) per chunk. Keeps peak memory to
            `stride` windows instead of all of them."""

            def __init__(self, feeds):
                self.feeds = feeds
                self.start = 0
                self.end = len(feeds)
                self.i = 0

            def set_range(self, start_index, end_index):
                self.start = start_index
                self.end = end_index
                self.i = start_index

            def get_next(self):
                if self.i >= self.end:
                    return None
                f = self.feeds[self.i]
                self.i += 1
                return f

            def __len__(self):
                return len(self.feeds)

        method = {
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile,
        }[args.calib_method]
        print(
            f"quantizing (STATIC PTQ, QDQ, per_channel={args.per_channel}, "
            f"method={args.calib_method}) ..."
        )
        if op_types:
            print(f"  quantizing only op types: {op_types}")
        else:
            print("  quantizing all supported ops (Conv + MatMul + Gemm)")
        # Always use the manual staged path: ORT's quantize_static wrapper OOMs
        # on this model (its CalibStridedMinMax/deepcopy handling), while the
        # staged augment->forward->compute->quantize peaks at a safe ~4GB.
        _staged_static(to_quantize, args.quant_out, Reader(feeds), method, args)
    else:
        from onnxruntime.quantization import quantize_dynamic

        print(f"quantizing (dynamic, per_channel={args.per_channel}) ...")
        kwargs = dict(weight_type=QuantType.QInt8, per_channel=args.per_channel)
        if op_types:
            kwargs["op_types_to_quantize"] = op_types
            print(f"  quantizing only op types: {op_types}")
        quantize_dynamic(to_quantize, args.quant_out, **kwargs)

    print(f"wrote {args.quant_out}")
    print(f"  fp32 {src}: {_mb(src):.0f} MB  ->  int8 {args.quant_out}: "
          f"{_mb(args.quant_out):.0f} MB")
    Path(pre).unlink(missing_ok=True)


def make_session(args):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    # Graph optimization and the memory-arena/mem-pattern planner are INDEPENDENT
    # knobs. --lean turns off the arena + mem-pattern (the real RAM savers on the
    # 4GB Pi) but must NOT drop the optimization level: int8 (QDQ) models need
    # full optimization to fuse Quantize/Dequantize into QLinearConv/QLinearMatMul,
    # or you pay Q/DQ overhead with no int8 speedup. So keep opt at the chosen
    # level regardless of --lean.
    levels = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }
    so.graph_optimization_level = levels[args.opt_level]
    if args.lean:
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    return ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])


def cmd_bench(args):
    """Time the ORT core forward, framed like profile_demucs.py (per-forward ms +
    RTF on one 7.8s window). With --torch-baseline, also time the full torch
    forward on the same box for a same-machine ORT-vs-torch comparison."""
    import time

    import numpy as np

    window_s = args.window_seconds
    sess = make_session(args)

    # Derive input shapes from the ONNX graph so we DON'T need to load the torch
    # model (which would ~double peak RAM and OOM the 4GB Pi). Random inputs are
    # fine for timing -- compute is data-independent.
    feed = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        feed[inp.name] = np.random.randn(*shape).astype(np.float32)

    print(f"ORT core: threads={args.threads}  window={window_s:.3f}s")
    print(f"  peak RSS after session load: {_fmt_mb(peak_rss_mb())}")
    sess.run(None, feed)  # warmup
    t0 = time.perf_counter()
    for _ in range(args.runs):
        sess.run(None, feed)
    dt = (time.perf_counter() - t0) / args.runs
    rtf = dt / window_s
    print(
        f"  per-forward (core only): {dt*1000:.1f} ms  |  RTF {rtf:.2f}x "
        f"({'faster' if rtf < 1 else 'SLOWER'} than real-time)"
    )
    print(f"  peak RSS after forward: {_fmt_mb(peak_rss_mb())}")

    if args.torch_baseline:
        m = get_htdemucs(args.name)
        ctx = make_core_inputs(m)
        th.set_num_threads(args.threads)
        with th.no_grad():
            m(ctx["mix"])  # warmup
            t0 = time.perf_counter()
            for _ in range(args.runs):
                m(ctx["mix"])
            dt_t = (time.perf_counter() - t0) / args.runs
        rtf_t = dt_t / window_s
        print(
            f"  torch full forward:      {dt_t*1000:.1f} ms  |  RTF {rtf_t:.2f}x"
        )
        print(f"  ORT speedup vs torch: {dt_t/dt:.2f}x")


def cmd_e2e(args):
    """End-to-end: run demucs' real apply_model (windowing + overlap-add) on a
    dummy track, but with the heavy core in ORT. Gives a true track-level RTF
    directly comparable to profile_demucs.py --track-seconds. Loads the torch
    model for the STFT/mask helpers + config, so use --lean on the Pi."""
    import time

    from demucs.apply import apply_model

    m = get_htdemucs(args.name)
    print(f"peak RSS after torch model load: {_fmt_mb(peak_rss_mb())}")
    sess = make_session(args)
    model = OrtCoreModel(m, sess)
    print(f"peak RSS after ORT session: {_fmt_mb(peak_rss_mb())}")

    sr = m.samplerate
    ch = m.audio_channels
    length = int(args.track_seconds * sr)
    track = th.randn(1, ch, length)

    print(
        f"\n=== ORT apply_model on {args.track_seconds:.1f}s track "
        f"(threads={args.threads}, shifts={args.shifts}, overlap={args.overlap}) ==="
    )
    with th.no_grad():
        t0 = time.perf_counter()
        apply_model(
            model,
            track,
            shifts=args.shifts,
            split=True,
            overlap=args.overlap,
            device="cpu",
            progress=False,
        )
        dt = time.perf_counter() - t0

    rtf = dt / args.track_seconds
    print(
        f"  wall-clock: {dt:.1f}s for {args.track_seconds:.1f}s audio -> RTF "
        f"{rtf:.2f}x ({'faster' if rtf < 1 else 'SLOWER'} than real-time)"
    )
    print(f"  peak RSS during full run: {_fmt_mb(peak_rss_mb())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", default="htdemucs")
    ap.add_argument("-o", "--out", default="profile/htdemucs_core.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--audio", default="",
                    help="mp3 for --verify to use a real-audio window instead of "
                    "random noise (required for a fair int8 quality check)")
    ap.add_argument(
        "--window-seconds",
        type=float,
        default=7.8,
        help="segment length the core was exported at (htdemucs = 39/5 = 7.8s); "
        "only used to compute RTF in --bench",
    )
    ap.add_argument(
        "--torch-baseline",
        action="store_true",
        help="also time the full torch forward on this machine for comparison",
    )
    ap.add_argument(
        "--lean",
        action="store_true",
        help="low-RAM ORT session (no arena/mem-pattern) for the 4GB Pi; keeps "
        "full graph optimization so int8 QDQ still fuses",
    )
    ap.add_argument(
        "--opt-level", default="all",
        choices=["all", "extended", "basic", "disabled"],
        help="ORT graph optimization level; keep 'all' for int8 QDQ fusion",
    )
    ap.add_argument("--quant-out", default="profile/htdemucs_core_int8.onnx",
                    help="output path for --quantize")
    ap.add_argument("--per-channel", action="store_true",
                    help="per-channel weight quant (much better for conv)")
    ap.add_argument("--exclude-ops", default="",
                    help="comma-sep op types to NOT quantize, e.g. 'MatMul' to "
                    "keep the transformer/attention in fp32")
    ap.add_argument("--static", action="store_true", default=True,
                    help="static PTQ with calibration (default; right for CNNs)")
    ap.add_argument("--no-static", dest="static", action="store_false",
                    help="use dynamic quant instead (weights only, no calibration)")
    ap.add_argument("--calib-dir", default="profile/calibration_mp3",
                    help="folder of mp3s to calibrate static PTQ on")
    ap.add_argument("--calib-windows", type=int, default=64,
                    help="max 7.8s windows to calibrate on")
    ap.add_argument("--calib-method", default="minmax",
                    choices=["minmax", "entropy", "percentile"])
    ap.add_argument("--debug-stages", action="store_true",
                    help="run static PTQ stage-by-stage with peak-RSS prints to "
                    "locate the OOM")
    ap.add_argument("--calib-stride", type=int, default=1,
                    help="stream calibration in chunks of this many windows "
                    "(CalibStridedMinMax); keeps peak RAM to this many windows. "
                    "0 = collect all at once. minmax only. must divide "
                    "--calib-windows.")
    ap.add_argument("--track-seconds", type=float, default=60.0,
                    help="dummy track length for --e2e")
    ap.add_argument("--shifts", type=int, default=1, help="--e2e apply_model shifts")
    ap.add_argument("--overlap", type=float, default=0.25, help="--e2e overlap")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--bench", action="store_true")
    g.add_argument("--e2e", action="store_true")
    g.add_argument("--quantize", action="store_true")
    args = ap.parse_args()
    if args.export:
        cmd_export(args)
    elif args.verify:
        cmd_verify(args)
    elif args.bench:
        cmd_bench(args)
    elif args.quantize:
        cmd_quantize(args)
    else:
        cmd_e2e(args)


if __name__ == "__main__":
    main()
