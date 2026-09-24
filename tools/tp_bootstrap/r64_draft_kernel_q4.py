"""Experiment override: the r63 draft levers plus a Q4_G64_FP16 kernel projection.

r64 of the PLAN.md section 8 campaign. Each DFlash2 draft layer owns two dynamic-grouped-convolution
kernel projections [1280,5120] (attention_conv and mlp_conv) that the family recipe leaves at BF16:
_optional skips them. The prepare Op gained a Q4_G64_FP16 variant (a plain q4 linear profile for
[1280,5120] plus a BF16-input reduce), so this override moves both projections to Q4: ten objects,
about -92 MiB on shard 0.

The file is cumulative on top of r63: r57 levers, the r62 feature projection and the r63 finish
projections stay Q4, and only the ten kernel projections change. The official recipe keeps BF16; the
resulting artifact is an experiment piece only.

Apply with: tools.convert --override tools/tp_bootstrap/r64_draft_kernel_q4.py
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
KERNEL_ROLES = ("/attention_conv/kernel_projection", "/mlp_conv/kernel_projection")


def configure(model, recipe, sources):
    if FEATURE_PROJECTION not in model.parameters:
        raise ValueError(f"draft kernel q4 override: {FEATURE_PROJECTION} is not in the model")
    recipe.assign(FEATURE_PROJECTION, format=Q4, method=grouped_absmax)

    counts = dict.fromkeys(ROLES, 0)
    kernel_counts = dict.fromkeys(KERNEL_ROLES, 0)
    for name in model.parameters:
        if not name.startswith("dflash2/layers/"):
            continue
        kernel_role = next((suffix for suffix in KERNEL_ROLES if name.endswith(suffix)), None)
        if kernel_role is not None:
            kernel_counts[kernel_role] += 1
            recipe.assign(name, format=Q4, method=grouped_absmax)
            continue
        for suffix, qtype in ROLES.items():
            if name.endswith(suffix):
                recipe.assign(name, format=qtype, method=grouped_absmax)
                counts[suffix] += 1
                break
    for suffix in ROLES:
        if counts[suffix] != EXPECTED_LAYERS:
            raise ValueError(
                f"draft kernel q4 override matched {counts[suffix]} {suffix} parameters, "
                f"expected {EXPECTED_LAYERS}"
            )
    for suffix in KERNEL_ROLES:
        if kernel_counts[suffix] != EXPECTED_LAYERS:
            raise ValueError(
                f"draft kernel q4 override matched {kernel_counts[suffix]} {suffix} parameters, "
                f"expected {EXPECTED_LAYERS}"
            )
