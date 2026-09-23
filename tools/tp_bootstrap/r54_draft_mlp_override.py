"""Experiment override: quantize the DFlash2 draft gate/up SwiGLU to Q4_G64_FP16.

Why only gate/up: the draft's dynamic grouped-conv finish
(src/ops/wrapper/dynamic_grouped_conv.cpp:61-73, require_finish_projection_weight)
hard-requires Q8_G32_FP16/RowSplit for the projection it fuses with the conv delta,
which is the MLP down projection and the attention output projection. The fused
gate/up SwiGLU has no such constraint: linear_swiglu implements Q4_G64_FP16 for the
draft's [34816,5120] shape. So this override reassigns exactly the 5 fused gate/up
objects and leaves every other draft tensor at its stored format.

Apply with: tools.convert --override tools/tp_bootstrap/r54_draft_mlp_override.py
"""

from tools.convert.methods import grouped_absmax

Q4 = "q4_g64_fp16"

EXPECTED_LAYERS = 5


def configure(model, recipe, sources):
    touched = []
    for name in model.parameters:
        if not name.startswith("dflash2/layers/") or "/mlp/" not in name:
            continue
        role = name.rsplit("/", 1)[-1]
        if role not in ("gate", "up"):
            continue
        recipe.assign(name, format=Q4, method=grouped_absmax)
        touched.append(name)
    for role in ("gate", "up"):
        count = sum(1 for name in touched if name.endswith("/" + role))
        if count != EXPECTED_LAYERS:
            raise ValueError(
                f"draft gate/up override matched {count} {role} parameters, expected {EXPECTED_LAYERS}"
            )
