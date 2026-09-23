"""Dry run the DFlash2 draft-MLP override: print resolved selections, no conversion."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.__main__ import _function  # noqa: E402
from tools.convert.qwen3_5 import build_model  # noqa: E402
from tools.convert.recipe import Recipe  # noqa: E402
from tools.convert.sources.safetensors import SafetensorsSource  # noqa: E402

BASE = Path(r"D:/LLM/W4A16/NVFP4/W4A4+W8A8")
DRAFT = BASE / "DFlash2-FP8"
USER_RECIPE = Path(r"D:/LLM/w4a4_family_recipe.py")
OVERRIDE = Path(__file__).with_name("r54_draft_mlp_override.py")


def main() -> None:
    with SafetensorsSource(BASE) as base, SafetensorsSource(DRAFT) as dflash2:
        sources = {"base": base, "quantized": base, "dflash2": dflash2}
        model = build_model(
            base,
            components=("text", "vision", "mtp", "dflash2"),
            companions={"dflash2": dflash2},
        )
        recipe = Recipe(model)
        _function(str(USER_RECIPE))(model, recipe, sources)
        _function(str(OVERRIDE))(model, recipe, sources)
        for name in sorted(recipe.selections):
            if not name.startswith("dflash2/layers/"):
                continue
            for selection in recipe.selections[name]:
                method = getattr(selection.method, "__name__", selection.method)
                print(
                    f"{name} [{selection.begin},{selection.end}) "
                    f"{selection.format} {method} src={selection.source.label}"
                )


if __name__ == "__main__":
    main()
