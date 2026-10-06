from __future__ import annotations

import torch

from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import assign_dflash_formats
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source


CODEBOOKS = (
    "dflash2/candidate_selector/predecessor_codebook",
    "dflash2/candidate_selector/successor_codebook",
)


def _parameter(name: str) -> Parameter:
    """A projection, except for the codebooks, which the model declares as direct parents."""

    values = torch.ones(4, 128, dtype=torch.bfloat16)
    inputs = () if name in CODEBOOKS else (name + "_input",)
    return Parameter(name, (4, 128), array_source(values, name), inputs=inputs)


def _formats(names) -> dict:
    model = Model({"text": {"config": {}}})
    for name in names:
        model.add(_parameter(name))
    recipe = Recipe(model)
    assign_dflash_formats(model, recipe)
    return {name: recipe.selections[name][0].format for name in model.parameters}


def test_dflash2_projections_quantize_and_codebooks_stay_unquantized():
    formats = _formats((
        "dflash2/feature_projection",
        "dflash2/layers/0/attention/query",
        "dflash2/layers/0/attention/output",
        "dflash2/layers/0/mlp/gate",
        "dflash2/layers/0/mlp/down",
        "dflash2/layers/0/attention_conv/kernel_projection",
        "dflash2/layers/0/mlp_conv/kernel_projection",
        "dflash2/candidate_selector/hidden_projection",
        "dflash2/candidate_selector/predecessor_codebook",
        "dflash2/candidate_selector/successor_codebook",
        "mtp/input_projection",
    ))
    for name in (
        "dflash2/feature_projection",
        "dflash2/layers/0/attention/query",
        "dflash2/layers/0/attention/output",
        "dflash2/layers/0/mlp/gate",
        "dflash2/layers/0/mlp/down",
        "dflash2/layers/0/attention_conv/kernel_projection",
        "dflash2/layers/0/mlp_conv/kernel_projection",
    ):
        assert formats[name] == "q4_g64_fp16", name
    # The whole-vocabulary codebooks and the selector projection stay direct, not grouped.
    assert formats["dflash2/candidate_selector/hidden_projection"] == "bf16"
    assert formats["dflash2/candidate_selector/predecessor_codebook"] == "bf16"
    assert formats["dflash2/candidate_selector/successor_codebook"] == "bf16"
    # DFlash v1 and MTP keep Q8.
    assert formats["mtp/input_projection"] == "q8_g32_fp16"


def test_codebook_assignment_is_a_direct_cast_not_a_quantizer():
    model = Model({"text": {"config": {}}})
    name = "dflash2/candidate_selector/predecessor_codebook"
    model.add(_parameter(name))
    recipe = Recipe(model)
    assign_dflash_formats(model, recipe)
    selection = recipe.selections[name][0]
    assert selection.format == "bf16"
    assert selection.method is not None and selection.method.__name__ == "cast_direct"
