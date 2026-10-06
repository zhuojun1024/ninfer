from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.graft_bindings import main

_TEMPLATE = b"custom {{ messages }}"
_NORM = struct.pack("<4H", 0x3C00, 0x4000, 0x4200, 0x4400)
_OVERRIDE_NORM = struct.pack("<4H", 0x0001, 0x0002, 0x0003, 0x0004)
_WEIGHT_BF16 = struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080)
_WEIGHT_FP32 = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)


def _fixture(path: Path, *, weight_format: str, weight_words: bytes,
             norm_words: bytes = _NORM, include_weight: bool = True) -> None:
    """One tiny text-only artifact whose weight binding can carry either representation."""

    specs = [
        TensorSpec("w", (2, 2), weight_format, "contiguous_le_v1"),
        TensorSpec("norm", (4,), "bf16", "contiguous_le_v1"),
        TensorSpec("scale", (), "fp32", "contiguous_le_v1"),
        ResourceSpec("template", len(_TEMPLATE)),
    ]
    components = {
        "text": {
            "config": {"architectures": ["Qwen3_5ForCausalLM"], "hidden_size": 2},
            "resources": {"chat_template.jinja": "template"},
        }
    }
    bindings = {
        "text/view": {
            "parts": [
                {"object": "w", "range": [0, 2]},
                {"object": "w", "range": [2, 4]},
            ]
        },
        "text/norm": {"object": "norm"},
    }
    if include_weight:
        bindings["text/weight"] = {"object": "w"}
    uses = []
    if include_weight:
        uses.append(
            {
                "parameter": "text/weight",
                "input": "text/input",
                "activation_policy": "AllowA4",
                "auxiliaries": {"activation_input_divisor": {"object": "scale"}},
            }
        )
    with ArtifactWriter(path, specs, components=components, bindings=bindings, uses=uses,
                        metadata={"name": "fixture"}) as writer:
        writer.write_object("w", weight_words)
        writer.write_object("norm", norm_words)
        writer.write_object("scale", struct.pack("<f", 2.0))
        writer.write_object("template", _TEMPLATE)


def _pair(tmp_path: Path):
    source = tmp_path / "source.ninfer"
    override = tmp_path / "override.ninfer"
    _fixture(source, weight_format="bf16", weight_words=_WEIGHT_BF16)
    _fixture(override, weight_format="fp32", weight_words=_WEIGHT_FP32,
             norm_words=_OVERRIDE_NORM)
    return source, override


def test_replaces_selected_binding_and_keeps_everything_else(tmp_path):
    source, override = _pair(tmp_path)
    out = tmp_path / "out.ninfer"

    assert main(["--source", str(source), "--override", str(override),
                 "--binding", "text/weight", "--name", "mixed", "--out", str(out)]) == 0

    written = Artifact(out)
    weight = written.by_id[written.directory.bindings["text/weight"]["object"]]
    assert weight.format == "fp32" and int(weight.bytes) == 16
    assert written.read_object(weight.id) == _WEIGHT_FP32

    # An unselected binding keeps the source representation and bytes.
    norm = written.by_id[written.directory.bindings["text/norm"]["object"]]
    assert norm.format == "bf16" and written.read_object(norm.id) == _NORM

    # An unselected view of the same source object keeps the source bytes and ranges: the rebuilt
    # document does not alias it to the replaced copy.
    view = written.directory.bindings["text/view"]
    assert [part["range"] for part in view["parts"]] == [[0, 2], [2, 4]]
    assert written.read_object(view["parts"][0]["object"]) == _WEIGHT_BF16
    assert view["parts"][0]["object"] != written.directory.bindings["text/weight"]["object"]

    # Component resources and Use auxiliaries survive the rebuilt document.
    resource = written.directory.components["text"]["resources"]["chat_template.jinja"]
    assert written.read_object(resource) == _TEMPLATE
    auxiliary = written.directory.uses[0]["auxiliaries"]["activation_input_divisor"]["object"]
    assert written.read_object(auxiliary) == struct.pack("<f", 2.0)

    assert written.directory.metadata["name"] == "mixed"
    assert written.directory.provenance["replaced_bindings"] == ["text/weight"]
    assert written.directory.provenance["sources"]["override"]["path"] == str(override)


def test_pattern_selects_every_match(tmp_path):
    source, override = _pair(tmp_path)
    out = tmp_path / "out.ninfer"
    assert main(["--source", str(source), "--override", str(override),
                 "--binding", "text/*", "--out", str(out)]) == 0
    written = Artifact(out)
    assert set(written.directory.provenance["replaced_bindings"]) == {
        "text/norm", "text/view", "text/weight"}
    # text/norm is matched by the pattern too, so it must take the override's bytes.
    norm = written.by_id[written.directory.bindings["text/norm"]["object"]]
    assert written.read_object(norm.id) == _OVERRIDE_NORM


def test_dry_run_writes_nothing(tmp_path):
    source, override = _pair(tmp_path)
    out = tmp_path / "out.ninfer"
    assert main(["--source", str(source), "--override", str(override),
                 "--binding", "text/weight", "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_rejects_unmatched_and_absent_bindings(tmp_path):
    source, override = _pair(tmp_path)
    out = tmp_path / "out.ninfer"
    with pytest.raises(SystemExit):
        main(["--source", str(source), "--override", str(override),
              "--binding", "text/missing", "--out", str(out)])
    absent = tmp_path / "absent.ninfer"
    _fixture(absent, weight_format="fp32", weight_words=struct.pack("<4f", 0.0, 0.0, 0.0, 0.0),
             include_weight=False)
    with pytest.raises(SystemExit):
        main(["--source", str(source), "--override", str(absent),
              "--binding", "text/weight", "--out", str(out)])
    assert not out.exists()


def test_requires_force_to_replace_an_existing_output(tmp_path):
    source, override = _pair(tmp_path)
    out = tmp_path / "out.ninfer"
    out.write_bytes(b"stale")
    with pytest.raises(SystemExit):
        main(["--source", str(source), "--override", str(override),
              "--binding", "text/weight", "--out", str(out)])
    assert out.read_bytes() == b"stale"
    assert main(["--source", str(source), "--override", str(override),
                 "--binding", "text/weight", "--out", str(out), "--force"]) == 0
    assert Artifact(out).directory.provenance["replaced_bindings"] == ["text/weight"]
