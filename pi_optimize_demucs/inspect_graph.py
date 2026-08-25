"""Inspect an exported/quantized core: op-type census + shapes of hot nodes.

Answers two questions the ORT profile raised:
  1. Is this really STATIC int8? Static QDQ fuses to QLinearConv. Seeing
     ConvInteger + DynamicQuantizeLinear means DYNAMIC quant, which on ARM
     is far slower and re-quantizes activations every single forward.
  2. Why is one conv eating most of the runtime? Print its attributes.
"""

import sys
from collections import Counter

import onnx
from onnx import shape_inference

path = sys.argv[1]
focus = sys.argv[2] if len(sys.argv) > 2 else None

model = onnx.load(path)
g = model.graph

census = Counter(n.op_type for n in g.node)
print(f"{path}\nnodes={len(g.node)}\n")
print("op-type census (top 20):")
for op, n in census.most_common(20):
    print(f"  {n:5d}  {op}")

quant_style = []
if census.get("ConvInteger") or census.get("DynamicQuantizeLinear"):
    quant_style.append("DYNAMIC (ConvInteger/DynamicQuantizeLinear)")
if census.get("QLinearConv"):
    quant_style.append("STATIC QDQ fused (QLinearConv)")
if census.get("QuantizeLinear") and not census.get("QLinearConv"):
    quant_style.append("QDQ present but NOT fused to QLinearConv")
print("\nquantization style:", " + ".join(quant_style) or "fp32 (no quant ops)")

# shapes
inferred = shape_inference.infer_shapes(model)
vi = {
    v.name: v for v in list(inferred.graph.value_info) + list(g.input) + list(g.output)
}


def shape_of(name):
    v = vi.get(name)
    if v is None:
        return "?"
    d = v.type.tensor_type.shape.dim
    return [x.dim_value if x.dim_value else (x.dim_param or "?") for x in d]


inits = {i.name: i for i in g.initializer}

if focus:
    print(f"\n--- nodes matching {focus!r} ---")
    for n in g.node:
        if focus not in n.name:
            continue
        print(f"\nnode: {n.name}\n  op: {n.op_type}")
        for a in n.attribute:
            val = onnx.helper.get_attribute_value(a)
            print(f"  attr {a.name}: {val}")
        for i, inp in enumerate(n.input):
            extra = ""
            if inp in inits:
                dims = list(inits[inp].dims)
                extra = f"  (initializer dims={dims})"
            print(f"  in[{i}]  {inp}  shape={shape_of(inp)}{extra}")
        for i, out in enumerate(n.output):
            print(f"  out[{i}] {out}  shape={shape_of(out)}")
