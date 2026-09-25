"""Experiment override: the DFlash2 selector codebooks to Q4_G64_FP16.

r68 of the PLAN.md section 3.9 campaign (lever 1, the selector codebook). It is cumulative over r66
(all earlier Q4 draft levers) and deliberately does NOT include the r67 QKV change: r67 measured a
2 pp acceptance loss for 79.7 MiB and is not being adopted, so this artifact isolates the codebook.

The selector codebooks are one complete [vocab, rank] parent each, so the two assignments move
exactly two stored objects. The loader binds whichever representation the artifact carries (see
src/models/qwen3_5/execution/parameters.cpp and selector.h).

Apply with: tools.convert --override tools/tp_bootstrap/r68_draft_codebook_q4.py
"""

import importlib.util
import pathlib

from tools.convert.methods import grouped_absmax

Q4 = "q4_g64_fp16"
CODEBOOKS = (
    "dflash2/candidate_selector/predecessor_codebook",
    "dflash2/candidate_selector/successor_codebook",
)

_R66 = pathlib.Path(__file__).resolve().with_name("r66_draft_q4_all.py")
_SPEC = importlib.util.spec_from_file_location("_r68_r66_draft_q4_all", _R66)
_R66_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_R66_MODULE)


def configure(model, recipe, sources):
    _R66_MODULE.configure(model, recipe, sources)

    for name in CODEBOOKS:
        if name not in model.parameters:
            raise ValueError(f"r68 codebook q4 override: missing {name}")
        recipe.assign(name, format=Q4, method=grouped_absmax)
