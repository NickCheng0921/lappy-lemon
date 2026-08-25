"""Freeze input shapes and constant-fold the exported core.

Why this matters more than it looks: the torch export turned every einops
`rearrange` into runtime shape arithmetic -- the fp32 graph carries 165 Shape,
88 Gather and 277 Reshape nodes. Downstream, ORT cannot infer static shapes,
so tensors show up as `unk__588 x 192 x unk__590` and conv kernels fall back to
generic paths. On the Pi that made ONE 1.2 GMAC conv (0.9% of the model's math)
take 81% of total runtime.

Freezing the (already fixed) input dims lets the folder evaluate all that shape
arithmetic at build time and hand ORT a fully static graph.

Run:
    python simplify.py in.onnx out.onnx
"""

import argparse
import os
import sys

import onnx


def count_dynamic(model):
    """How many value_info tensors still lack a fully static shape."""
    from onnx import shape_inference

    m = shape_inference.infer_shapes(model)
    dyn = static = 0
    for v in m.graph.value_info:
        dims = v.type.tensor_type.shape.dim
        if not dims:
            dyn += 1
            continue
        if all(d.dim_value for d in dims):
            static += 1
        else:
            dyn += 1
    return static, dyn


def census(model):
    from collections import Counter

    return Counter(n.op_type for n in model.graph.node)


def report(tag, model):
    c = census(model)
    st, dy = count_dynamic(model)
    shapey = sum(
        c.get(k, 0)
        for k in (
            "Shape",
            "Gather",
            "Unsqueeze",
            "Reshape",
            "Concat",
            "Slice",
            "Expand",
            "Constant",
        )
    )
    print(
        f"[{tag}] nodes={len(model.graph.node):5d}  "
        f"shape-arith nodes={shapey:4d}  "
        f"static tensors={st:4d}  dynamic={dy:4d}"
    )
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--backend", default="onnxslim", choices=["onnxslim", "onnxsim"])
    args = ap.parse_args()

    model = onnx.load(args.src)
    print(f"loaded {args.src} ({os.path.getsize(args.src)/1e6:.0f} MB)")

    # The exported core already has fully fixed input dims; make that explicit
    # so the folder can propagate them.
    shapes = {}
    for i in model.graph.input:
        dims = [
            d.dim_value if d.dim_value else None for d in i.type.tensor_type.shape.dim
        ]
        shapes[i.name] = dims
        print(f"  input {i.name}: {dims}")
        if any(d is None for d in dims):
            print("   !! dynamic input dim -- freeze it before simplifying")

    before = report("before", model)

    if args.backend == "onnxslim":
        import onnxslim

        out = onnxslim.slim(model)
    else:
        from onnxsim import simplify

        out, ok = simplify(model, overwrite_input_shapes=shapes)
        if not ok:
            print("!! onnxsim reported the simplified model did NOT validate")
            sys.exit(1)

    after = report("after ", out)

    onnx.save(out, args.dst, save_as_external_data=False)
    print(f"\nwrote {args.dst} ({os.path.getsize(args.dst)/1e6:.0f} MB)")

    print("\nop-type deltas (before -> after):")
    for op in sorted(set(before) | set(after), key=lambda o: -(before.get(o, 0))):
        b, a = before.get(op, 0), after.get(op, 0)
        if b != a:
            print(f"  {op:<24}{b:5d} -> {a:5d}   ({a-b:+d})")


if __name__ == "__main__":
    main()
