"""Dry run a DFlash2 draft-quantization override: print the resolved draft selections.

Usage: python tools/tp_bootstrap/r62_draft_q4_dryrun.py <override.py> [<override.py> ...]

Every override is applied on top of the user recipe in the order given, exactly as
tools.convert applies a single one, and the resolved dflash2 selections are printed with
their format counts. Nothing is written, which is what makes this the cheap preflight for
an override whose object count matters (r62-r66 of PLAN.md section 8).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.__main__ import _function  # noqa: E402
from tools.convert.qwen3_5 import build_model  # noqa: E402
from tools.convert.recipe import Recipe  # noqa: E402
from tools.convert.sources.safetensors import SafetensorsSource  # noqa: E402

DEFAULT_BASE = Path(r"D:/LLM/W4A16/NVFP4/W4A4+W8A8")
USER_RECIPE = Path(r"D:/LLM/w4a4_family_recipe.py")


def parse_arguments(arguments):
    """Return (base, draft, overrides); the draft defaults to <base>/DFlash2-FP8."""
    base = DEFAULT_BASE
    draft = None
    overrides = []
    index = 0
    while index < len(arguments):
        if arguments[index] == "--base":
            base = Path(arguments[index + 1])
            index += 2
        elif arguments[index] == "--draft":
            draft = Path(arguments[index + 1])
            index += 2
        else:
            overrides.append(Path(arguments[index]))
            index += 1
    return base, draft or (base / "DFlash2-FP8"), overrides


def main() -> None:
    base_path, draft_path, overrides = parse_arguments(sys.argv[1:])
    if not overrides:
        raise SystemExit(
            "usage: r62_draft_q4_dryrun.py [--base DIR] [--draft DIR] <override.py> [...]"
        )

    with SafetensorsSource(base_path) as base, SafetensorsSource(draft_path) as dflash2:
        sources = {"base": base, "quantized": base, "dflash2": dflash2}
        model = build_model(
            base,
            components=("text", "vision", "mtp", "dflash2"),
            companions={"dflash2": dflash2},
        )
        recipe = Recipe(model)
        _function(str(USER_RECIPE))(model, recipe, sources)
        for override in overrides:
            _function(str(override))(model, recipe, sources)

        formats = Counter()
        rows = []
        for name in sorted(recipe.selections):
            if not name.startswith("dflash2/"):
                continue
            for selection in recipe.selections[name]:
                method = getattr(selection.method, "__name__", selection.method)
                formats[selection.format] += 1
                rows.append(
                    f"{name} [{selection.begin},{selection.end}) "
                    f"{selection.format} {method} src={selection.source.label}"
                )
        for row in rows:
            print(row)
        print("--- dflash2 selections by format ---")
        for fmt, count in sorted(formats.items()):
            print(f"{fmt}: {count}")


if __name__ == "__main__":
    main()
