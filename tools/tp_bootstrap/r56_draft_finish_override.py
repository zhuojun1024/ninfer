"""Experiment override: quantize the DFlash2 draft finish projections to Q5_G64_FP16.

The draft's dynamic grouped-conv finish (linear_dynamic_grouped_conv_add) fuses its
projection with the conv delta. That op accepted only Q8_G32_FP16 RowSplit until the
r56 Q5 route (src/ops/dynamic_grouped_conv/q5/), so the five mlp/down [5120,17408]
and five attention/output [5120,4096] parameters were pinned Q8 by the family
recipe. With the Q5 finish route landed, this override reassigns exactly those ten
parameters to Q5_G64_FP16: about -214 MiB on shard 0 (down -172, output -42 across
5 layers). The official recipe keeps Q8; the resulting artifact is an experiment
piece only.

Apply with: tools.convert --override tools/tp_bootstrap/r56_draft_finish_override.py
"""

from tools.convert.methods import grouped_absmax

Q5 = "q5_g64_fp16"

EXPECTED_LAYERS = 5


def configure(model, recipe, sources):
    touched = []
    for name in model.parameters:
        if not name.startswith("dflash2/layers/"):
            continue
        if not (name.endswith("/attention/output") or name.endswith("/mlp/down")):
            continue
        recipe.assign(name, format=Q5, method=grouped_absmax)
        touched.append(name)
    for suffix in ("/attention/output", "/mlp/down"):
        count = sum(1 for name in touched if name.endswith(suffix))
        if count != EXPECTED_LAYERS:
            raise ValueError(
                f"draft finish override matched {count} {suffix} parameters, "
                f"expected {EXPECTED_LAYERS}"
            )
