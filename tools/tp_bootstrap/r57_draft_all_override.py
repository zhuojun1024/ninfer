"""Experiment override: apply every verified draft quantization lever at once.

Combines the three opt-in draft quantizations measured in PLAN.md section 3.7:
  * r54: dflash2/layers/*/mlp/{gate,up}    [34816, 5120] -> Q4_G64_FP16  (-450 MiB shard 0)
  * r55: dflash2/feature_projection        [5120, 25600] -> Q5_G64_FP16  ( -51 MiB shard 0)
  * r56: dflash2/layers/*/attention/output [5120,  4096] -> Q5_G64_FP16  ( -41 MiB shard 0)
         dflash2/layers/*/mlp/down         [5120, 17408] -> Q5_G64_FP16  (-172 MiB shard 0)

About -716 MiB on shard 0 versus the Q8-draft baseline. The official recipe keeps
Q8; the resulting artifact is an experiment piece only.

Apply with: tools.convert --override tools/tp_bootstrap/r57_draft_all_override.py
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
        raise ValueError(f"draft all-quant override: {FEATURE_PROJECTION} is not in the model")
    recipe.assign(FEATURE_PROJECTION, format=Q5, method=grouped_absmax)

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
                f"draft all-quant override matched {counts[suffix]} {suffix} parameters, "
                f"expected {EXPECTED_LAYERS}"
            )
