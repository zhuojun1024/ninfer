"""Experiment override: the DFlash2 draft attention QKV parent to Q4_G64_FP16.

r67 of the PLAN.md section 3.9 campaign. It is cumulative over r66 (all earlier Q4 draft
levers), so the produced artifact differs from the r66 q4all artifact only by the five fused
[6144,5120] QKV parents.

The draft's attention/{query,key,value} parameters form one packing group per layer, so the
QKV piece promotes exactly five stored objects. context_key/context_value are recipe shares of
key/value and therefore follow the same object; the decode and append paths project through the
row views when the parent is not Q8 (see src/models/qwen3_5/execution/draft.cpp).

The official recipe keeps Q8; the resulting artifact is an experiment piece only.

Apply with: tools.convert --override tools/tp_bootstrap/r67_draft_qkv_q4.py
"""

import importlib.util
import pathlib

from tools.convert.methods import grouped_absmax

Q4 = "q4_g64_fp16"

EXPECTED_LAYERS = 5
ROLES = ("query", "key", "value")

_R66 = pathlib.Path(__file__).resolve().with_name("r66_draft_q4_all.py")
_SPEC = importlib.util.spec_from_file_location("_r67_r66_draft_q4_all", _R66)
_R66_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_R66_MODULE)


def configure(model, recipe, sources):
    _R66_MODULE.configure(model, recipe, sources)

    counts = dict.fromkeys(ROLES, 0)
    for name in model.parameters:
        if not name.startswith("dflash2/layers/"):
            continue
        for role in ROLES:
            if name.endswith("/attention/" + role):
                recipe.assign(name, format=Q4, method=grouped_absmax)
                counts[role] += 1
                break
    for role in ROLES:
        if counts[role] != EXPECTED_LAYERS:
            raise ValueError(
                f"r67 draft qkv q4 override matched {counts[role]} attention/{role} "
                f"parameters, expected {EXPECTED_LAYERS}"
            )
