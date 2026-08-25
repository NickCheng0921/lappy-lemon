"""Memory-bounded static PTQ (QDQ int8) for the htdemucs core.

Why the previous attempt OOM'd
------------------------------
`create_calibrator(..., op_types_to_calibrate=None)` augments EVERY tensor in
the graph into a model output. Even though MinMax reduces each to a scalar, ORT
can no longer free or reuse any intermediate buffer -- the memory planner is
defeated -- and this model's spec-branch activations are ~132 MB each
([1,48,2048,336] fp32). Peak blows past 19 GB and the process is killed.

Fix: only the ops we actually quantize need ranges. Restricting
op_types_to_calibrate to Conv/Gemm/MatMul/ConvTranspose cuts the augmented
outputs by ~10x and lets the planner reuse buffers again.

Also streams calibration windows instead of materialising them all (each
window is ~14 MB of input alone).

Run:
    python quantize_static.py --src models/core_slim.onnx \
                              --dst models/core_slim_int8.onnx \
                              --windows 8
"""

import argparse
import gc
import resource
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor"))
sys.path.insert(0, str(REPO / "profile"))

import onnx

DEFAULT_CALIB_OPS = ["Conv", "ConvTranspose", "Gemm", "MatMul"]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def stage(msg):
    print(f"  [stage] {msg}: peak RSS {rss_mb():.0f} MB", flush=True)


class StreamingReader:
    """CalibrationDataReader that generates windows lazily.

    Holding all windows costs ~14 MB each; generating on demand keeps only one
    alive. Supports set_range() so ORT's strided calibration can walk chunks.
    """

    def __init__(self, make_iter, total):
        self.make_iter = make_iter
        self.total = total
        self._it = None
        self.start, self.end, self.i = 0, total, 0

    def set_range(self, start_index, end_index):
        self.start, self.end, self.i = start_index, end_index, start_index
        self._it = None

    def get_next(self):
        if self.i >= self.end:
            return None
        if self._it is None:
            self._it = self.make_iter()
            for _ in range(self.start):  # skip to window `start`
                next(self._it, None)
        item = next(self._it, None)
        if item is None:
            return None
        self.i += 1
        return item

    def __len__(self):
        return self.total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--calib-dir", default="profile/calibration_mp3")
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    ap.add_argument(
        "--calib-ops",
        default=",".join(DEFAULT_CALIB_OPS),
        help="op types to calibrate AND quantize",
    )
    ap.add_argument(
        "--exclude-nodes",
        default="",
        help="comma-sep substrings; matching nodes stay fp32. Use to "
        "rescue any op that quantizes badly (e.g. 'tdecoder.1')",
    )
    ap.add_argument(
        "--symmetric",
        action="store_true",
        help="symmetric activation ranges (sometimes better for conv)",
    )
    ap.add_argument(
        "--act-type",
        default="uint8",
        choices=["uint8", "int8"],
        help="activation dtype. uint8 is ORT's default and the "
        "combo ARM QLinearConv kernels are tuned for (u8 act + s8 "
        "weight); int8 activations are both slower and less accurate.",
    )
    ap.add_argument(
        "--calib-method",
        default="minmax",
        choices=["minmax", "percentile", "entropy"],
        help="minmax is wrecked by transients -- one loud sample "
        "sets a coarse scale for the whole tensor. percentile clips "
        "outliers and usually recovers most of the lost SNR.",
    )
    args = ap.parse_args()

    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization.calibrate import CalibrationMethod, create_calibrator
    from onnxruntime.quantization.qdq_quantizer import QDQQuantizer

    from onnx_export import get_htdemucs, _iter_core_feeds

    calib_ops = [s for s in args.calib_ops.split(",") if s]
    print(f"src={args.src}\ndst={args.dst}")
    print(
        f"calibrating op types: {calib_ops}  (restricting these is what "
        f"keeps peak RAM bounded)"
    )

    m = get_htdemucs()
    stage("after torch model load")

    def make_iter():
        return _iter_core_feeds(m, args.calib_dir, args.windows)

    reader = StreamingReader(make_iter, args.windows)

    method = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }[args.calib_method]
    extra = {"symmetric": args.symmetric}
    if args.calib_method == "percentile":
        # 99.999th pct: clip the extreme transient tail that makes minmax
        # scales useless, without cutting real signal.
        extra["percentile"] = 99.999
        extra["num_bins"] = 2048

    aug = args.dst + ".augmented.onnx"
    calibrator = create_calibrator(
        Path(args.src),
        calib_ops,
        augmented_model_path=aug,
        calibrate_method=method,
        extra_options=extra,
    )
    stage("after create_calibrator (augment)")

    calibrator.collect_data(reader)
    stage("after collect_data (calibration forwards)")

    tensors_range = calibrator.compute_data()
    stage("after compute_data (ranges)")
    del calibrator
    gc.collect()

    nodes_to_exclude = []
    if args.exclude_nodes:
        pats = [p for p in args.exclude_nodes.split(",") if p]
        model_peek = onnx.load(args.src, load_external_data=False)
        nodes_to_exclude = [
            n.name for n in model_peek.graph.node if any(p in n.name for p in pats)
        ]
        print(f"  excluding {len(nodes_to_exclude)} node(s) from quantization")
        del model_peek
        gc.collect()

    act_qtype = QuantType.QUInt8 if args.act_type == "uint8" else QuantType.QInt8
    print(f"  weights=QInt8  activations=Q{args.act_type.capitalize()}")

    model = onnx.load(args.src)
    # NOTE: QDQQuantizer's positional order is (weight_qType, activation_qType)
    # -- weights first. Easy to swap by accident.
    quantizer = QDQQuantizer(
        model,
        args.per_channel,
        False,  # reduce_range
        QuantType.QInt8,  # weight_qType
        act_qtype,  # activation_qType
        tensors_range,
        [],  # nodes_to_quantize (empty = all eligible)
        nodes_to_exclude,
        calib_ops,
        {"MatMulConstBOnly": False},
    )
    quantizer.quantize_model()
    stage("after quantize_model")

    quantizer.model.save_model_to_file(args.dst, False)
    stage("after save")
    Path(aug).unlink(missing_ok=True)

    import os

    print(
        f"\nwrote {args.dst} "
        f"({os.path.getsize(args.dst)/1e6:.0f} MB, "
        f"src was {os.path.getsize(args.src)/1e6:.0f} MB)"
    )

    from collections import Counter

    c = Counter(n.op_type for n in onnx.load(args.dst).graph.node)
    print(
        "quant ops:",
        {k: v for k, v in c.items() if "Quant" in k or "Linear" in k or "Integer" in k},
    )


if __name__ == "__main__":
    main()
