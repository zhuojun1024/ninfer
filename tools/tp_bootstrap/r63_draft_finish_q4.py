"""Experiment override: the r62 draft levers plus Q4_G64_FP16 finish projections.

r63 of the PLAN.md section 8 campaign. The DFlash2 draft MLP down [5120,17408] and attention
output [5120,4096] projections feed the fused linear_dynamic_grouped_conv_add, whose wrapper
admitted only Q5_G64_FP16 and Q8_G32_FP16. This step adds the Q4_G64_FP16 variant of that
fused Op (src/ops/dynamic_grouped_conv/q4/, backed by the new plain-linear Q4 profiles
n5120_k17408 and n5120_k4096) and points both projections at it.

The file is cumulative on top of r62: it re-applies the r57 levers with the r62 feature
projection and then moves the two finish projections from Q5 to Q4, so the produced artifact
differs from the r62 piece in exactly ten objects (5 down + 5 output). The official recipe keeps
Q8; the resulting artifact is an experiment piece only.

Apply with: tools.convert --override tools/tp_bootstrap/r63_draft_finish_q4.py
"""

from tools.convert.methods import grouped_absmax

Q4 = "q4_g64_fp16"

EXPECTED_LAYERS = 5
FEATURE_PROJECTION = "dflash2/feature_projection"

# Layer role suffixes stay disjoint so each parameter matches exactly one format.
ROLES = {
    "/mlp/gate": Q4,
    "/mlp/up": Q4,
    "/mlp/down": Q4,
    "/attention/output": Q4,
}


def configure(model, recipe, sources):
    if FEATURE_PROJECTION not in model.parameters:
        raise ValueError(f"draft finish q4 override: {FEATURE_PROJECTION} is not in the model")
    recipe.assign(FEATURE_PROJECTION, format=Q4, method=grouped_absmax)

    counts = dict.fromkeys(ROLES, 0)
    for name in model.parameters:
        if not name.startswith("dflash2/layers/"):
            continue
        for suffix, qtype in ROLES.items():
            if name.endswith(suffix):
                recipe.assign(name, format=qtype, method=grouped_absmax)
                counts[suffix] += 1
                break
    for suffix in ROLES:
        if counts[suffix] != EXPECTED_LAYERS:
            raise ValueError(
                f"draft finish q4 override matched {counts[suffix]} {suffix} parameters, "
                f"expected {EXPECTED_LAYERS}"
            )
