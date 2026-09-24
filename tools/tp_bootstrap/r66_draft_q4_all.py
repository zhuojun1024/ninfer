"""Experiment override: every accepted section 8 Q4 draft lever in one recipe.

r66 of the PLAN.md section 8 campaign. This is the merged piece the campaign ships: the r57 levers
plus the Q4_G64_FP16 draft projections that each earlier step qualified (r62 feature_projection,
r63 MLP down and attention output, r64 dynamic-grouped-conv kernel projections). It is cumulative
by construction and adds nothing itself; its job is the single count-checked recipe that produces
the campaign artifact, so the artifact identity claim is one file rather than a chain of imports.

Counted selections (parameter names): feature_projection 1, mlp/gate 5, mlp/up 5, mlp/down 5,
attention/output 5, attention_conv/kernel_projection 5, mlp_conv/kernel_projection 5 = 31. gate and
up share one stored object per layer, so the piece promotes 26 stored objects.

r65 (context_key) is deferred: the draft's attention/{query,key,value,context_key,context_value}
all alias one [6144,5120] object per layer, so it needs 4-bit variants of both the fused
attn_input_proj and the fused context_kv_materialize kernel, not another shape.

The official recipe keeps Q8/BF16 where this override changes them; the resulting artifact is an
experiment piece only.

Apply with: tools.convert --override tools/tp_bootstrap/r66_draft_q4_all.py
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
        raise ValueError(f"draft q4 all override: {FEATURE_PROJECTION} is not in the model")
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
                f"draft q4 all override matched {counts[suffix]} {suffix} parameters, "
                f"expected {EXPECTED_LAYERS}"
            )
    for suffix in KERNEL_ROLES:
        if kernel_counts[suffix] != EXPECTED_LAYERS:
            raise ValueError(
                f"draft q4 all override matched {kernel_counts[suffix]} {suffix} parameters, "
                f"expected {EXPECTED_LAYERS}"
            )
