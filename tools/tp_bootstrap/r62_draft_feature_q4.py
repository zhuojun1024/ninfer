"""Experiment override: the r57 draft levers plus a Q4_G64_FP16 feature projection.

r62 of the PLAN.md section 8 campaign. Section 3.7 pushed every DFlash2 draft tensor
down to the lowest qtype its consumer accepted, so the remaining space needs new op
variants. This step adds the plain-linear Q4 profile for [5120, 25600]
(src/ops/linear/q4/shapes/n5120_k25600.cu) and points the draft feature projection at it.

The file is cumulative: it re-applies the three verified section 3.7 levers (r54 gate/up
Q4, r55 feature projection Q5, r56 down/output Q5) so the produced artifact is the r57
draft-all piece with exactly one object changed and every A/B delta belongs to this step.
The official recipe keeps Q8; the resulting artifact is an experiment piece only.

Apply with: tools.convert --override tools/tp_bootstrap/r62_draft_feature_q4.py
"""

from tools.convert.methods import grouped_absmax

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"

EXPECTED_LAYERS = 5
FEATURE_PROJECTION = "dflash2/feature_projection"

# Layer role suffixes stay disjoint so each parameter matches exactly one format.
ROLES = {
    "/mlp/gate": Q4,
    "/mlp/up": Q4,
    "/mlp/down": Q5,
    "/attention/output": Q5,
}


def configure(model, recipe, sources):
    if FEATURE_PROJECTION not in model.parameters:
        raise ValueError(f"draft feature q4 override: {FEATURE_PROJECTION} is not in the model")
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
                f"draft feature q4 override matched {counts[suffix]} {suffix} parameters, "
                f"expected {EXPECTED_LAYERS}"
            )
