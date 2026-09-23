"""Experiment override: quantize the DFlash2 draft feature_projection to Q5_G64_FP16.

The draft's feature projection [5120, 25600] (5120 x 5 target-layer hidden
concatenation) is stored Q8 by the family recipe. The plain linear op gained a
Q5 profile for this exact shape (src/ops/linear/q5/shapes/n5120_k25600.cu), so
this override reassigns the single parameter to Q5_G64_FP16: about -63 MiB on
shard 0. The official recipe keeps Q8; the resulting artifact is an experiment
piece only.

Apply with: tools.convert --override tools/tp_bootstrap/r55_draft_feature_proj_override.py
"""

from tools.convert.methods import grouped_absmax

Q5 = "q5_g64_fp16"

NAME = "dflash2/feature_projection"


def configure(model, recipe, sources):
    if NAME not in model.parameters:
        raise ValueError(f"draft feature projection override: {NAME} is not in the model")
    recipe.assign(NAME, format=Q5, method=grouped_absmax)
