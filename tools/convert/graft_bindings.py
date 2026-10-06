"""Replace selected stored bindings of one v3 artifact with another artifact's bytes.

Every logical binding of a v3 artifact names the packed object that holds its tensor and the element
interval it views inside it, so a binding can be re-pointed at an object of a different
representation without decoding, re-quantizing or re-laying out any payload. Only the document and
the object offsets change.

This is the fine-grained companion of tools.convert.graft_components. A component graft keeps the
base artifact's document and overwrites objects in place, so it requires both artifacts to store
those objects with identical bytes, formats and view topology. A precision mix does not: restoring a
Q4_G64 selector codebook to the stored BF16 representation changes the object format, its byte
length, and - when the two artifacts alias rows into a shared parent differently - the binding's
view. Rebuilding the document handles all three.

Usage:
  python -m tools.convert.graft_bindings \
      --source GRAFTED.ninfer --override BASE.ninfer \
      --binding dflash2/candidate_selector/predecessor_codebook \
      --binding dflash2/candidate_selector/successor_codebook \
      --name rloo351-mixed-mtp-dflash2-vision-cbbf16 \
      --out MIXED.ninfer

--binding accepts an fnmatch pattern, so "--binding dflash2/layers/*/attention/context_key" selects
every layer. Everything the source artifact declares outside the selection is kept, including
component resources and the objects a Use auxiliary names. Pass --dry-run to print the plan without
writing.

The published output is reopened and checked before the command succeeds: every selected binding
must match the override artifact and every other binding must match the source artifact.
"""

from __future__ import annotations

import argparse
import fnmatch
from copy import deepcopy
from pathlib import Path

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter

MAX_FILE_BYTES = 32_000_000_000


def _refs(binding: dict) -> list[str]:
    """Object ids a binding references, in view order."""
    if "parts" in binding:
        return [str(part["object"]) for part in binding["parts"]]
    return [str(binding["object"])]


def _remap_binding(binding: dict, mapping: dict) -> dict:
    if "parts" in binding:
        return {"parts": [{**part, "object": mapping[str(part["object"])]}
                          for part in binding["parts"]]}
    return {"object": mapping[str(binding["object"])]}


def _remap_uses(use: dict, mapping: dict) -> dict:
    use = deepcopy(dict(use))
    if isinstance(use.get("auxiliaries"), dict):
        use["auxiliaries"] = {
            role: ({**value, "object": mapping[str(value["object"])]}
                   if isinstance(value, dict) and "object" in value else value)
            for role, value in use["auxiliaries"].items()
        }
    return use


def _keep_from_source(artifact: Artifact, selection: set) -> set:
    """Objects the output still needs from the source document."""

    kept: set = set()
    for name, binding in artifact.directory.bindings.items():
        if name not in selection:
            kept.update(_refs(binding))
    for component in artifact.directory.components.values():
        for value in (component.get("resources") or {}).values():
            if isinstance(value, str):
                kept.add(value)
            elif isinstance(value, dict) and "object" in value:
                kept.add(str(value["object"]))
    for use in artifact.directory.uses:
        for value in (use.get("auxiliaries") or {}).values():
            if isinstance(value, dict) and "object" in value:
                kept.add(str(value["object"]))
            elif isinstance(value, str):
                kept.add(value)
    return kept


def _signature(artifact: Artifact, binding: dict) -> list:
    """Comparable per-object view of one binding: representation, shape, bytes and range."""

    view = []
    for part in (binding["parts"] if "parts" in binding else [binding]):
        obj = artifact.by_id[str(part["object"])]
        view.append((
            getattr(obj, "format", None) or getattr(obj, "encoding", None),
            tuple(getattr(obj, "shape", ()) or ()),
            int(obj.bytes),
            tuple(part["range"]) if "range" in part else None,
        ))
    return view


def _select(artifact: Artifact, patterns: list) -> list:
    names: set = set()
    for pattern in patterns:
        matched = sorted(n for n in artifact.directory.bindings if fnmatch.fnmatchcase(n, pattern))
        if not matched:
            raise SystemExit(f"no binding in the source matches {pattern!r}")
        names.update(matched)
    return sorted(names)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="artifact that keeps its document and every unselected binding")
    parser.add_argument("--override", type=Path, required=True,
                        help="artifact read for the selected bindings' objects")
    parser.add_argument("--binding", action="append", default=[], required=True,
                        help="binding name or fnmatch pattern; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None, help="value stored in metadata.name")
    parser.add_argument("--force", action="store_true", help="replace an existing --out")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.out.resolve() in (args.source.resolve(), args.override.resolve()):
        raise SystemExit("--out must be a new path")
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to replace it")

    source = Artifact(args.source)
    override = Artifact(args.override)
    selection = _select(source, args.binding)
    missing = [name for name in selection if name not in override.directory.bindings]
    if missing:
        raise SystemExit(f"override artifact has no binding {missing[0]!r}")

    # Objects: everything the source keeps, plus the override's objects for the selection. An object
    # shared by a selected and an unselected binding is kept for the latter and the override's copy
    # is added, so the two views never alias.
    keep = _keep_from_source(source, set(selection))
    take = {object_id for name in selection
            for object_id in _refs(override.directory.bindings[name])}

    counter = [0]

    def new_id(kind: str) -> str:
        counter[0] += 1
        return f"weight/{counter[0]:06d}" if kind == "tensor" else f"resource/{counter[0]:06d}"

    def spec(object_id: str, obj):
        if getattr(obj, "kind", "tensor") == "tensor":
            return TensorSpec(object_id, obj.shape, obj.format, obj.layout)
        return ResourceSpec(object_id, obj.bytes, obj.encoding)

    specs = []
    plan = []
    source_map: dict = {}
    override_map: dict = {}
    for obj in source.directory.objects:
        if obj.id not in keep:
            continue
        object_id = new_id(obj.kind)
        source_map[obj.id] = object_id
        specs.append(spec(object_id, obj))
        plan.append((object_id, source, obj.offset, obj.bytes))
    for obj in override.directory.objects:
        if obj.id not in take:
            continue
        object_id = new_id(obj.kind)
        override_map[obj.id] = object_id
        specs.append(spec(object_id, obj))
        plan.append((object_id, override, obj.offset, obj.bytes))

    bindings = {
        name: (_remap_binding(override.directory.bindings[name], override_map)
               if name in selection else _remap_binding(binding, source_map))
        for name, binding in source.directory.bindings.items()
    }
    components = {}
    for name, component in source.directory.components.items():
        component = deepcopy(component)
        if isinstance(component.get("resources"), dict):
            component["resources"] = {
                role: (source_map[value] if isinstance(value, str)
                       else ({**value, "object": source_map[str(value["object"])]}
                             if isinstance(value, dict) and "object" in value else value))
                for role, value in component["resources"].items()
            }
        components[name] = component
    uses = [_remap_uses(use, source_map) for use in source.directory.uses]

    known = {entry.id for entry in specs}
    for name, binding in bindings.items():
        for object_id in _refs(binding):
            if object_id not in known:
                raise SystemExit(f"binding {name!r} references unknown object {object_id!r}")
    for use in uses:
        for value in (use.get("auxiliaries") or {}).values():
            if isinstance(value, dict) and "object" in value and str(value["object"]) not in known:
                raise SystemExit(f"Use auxiliary references unknown object {value['object']!r}")

    payload = sum(count for _, _, _, count in plan)
    print(f"source: {args.source.name} ({len(source.directory.objects)} objects)")
    print(f"override: {args.override.name} ({len(override.directory.objects)} objects)")
    print(f"selected {len(selection)} bindings: {', '.join(selection)}")
    print(f"objects: {len(specs)} (kept {len(source_map)} + override {len(override_map)})")
    print(f"bindings: {len(bindings)}  uses: {len(uses)}  components: {', '.join(components)}")
    print(f"payload: {payload:,} bytes ({payload / 2**30:.3f} GiB)")
    if args.dry_run:
        print("DRYRUN: plan validated, nothing written")
        return 0

    if args.out.exists():
        args.out.unlink()
    for stale in args.out.parent.glob(args.out.name + ".*.tmp"):
        stale.unlink()

    chunk = 16 * 1024 * 1024

    def stream(artifact: Artifact, offset: int, count: int):
        position = artifact.payload_offset + offset
        remaining = count
        with open(artifact.path, "rb") as handle:
            while remaining:
                size = min(chunk, remaining)
                handle.seek(position)
                data = handle.read(size)
                if len(data) < size:
                    raise SystemExit(f"short read {artifact.path}+{position}")
                yield data
                position += size
                remaining -= size

    provenance = {
        "converter": "byte-transplant",
        "tool": "tools/convert/graft_bindings.py",
        "sources": {
            "source": {"path": str(args.source), "artifact_id": source.artifact_id.hex()},
            "override": {"path": str(args.override), "artifact_id": override.artifact_id.hex()},
        },
        "replaced_bindings": selection,
        "policy": "every binding outside replaced_bindings is byte-for-byte from source",
    }
    metadata = dict(source.directory.metadata)
    if args.name:
        metadata["name"] = args.name
    with ArtifactWriter(args.out, specs, components=components, bindings=bindings, uses=uses,
                        metadata=metadata, provenance=provenance,
                        max_file_bytes=MAX_FILE_BYTES) as writer:
        for object_id, artifact, offset, count in plan:
            writer.write_object(object_id, stream(artifact, offset, count))

    written = Artifact(args.out)
    for name, binding in source.directory.bindings.items():
        expected = override if name in selection else source
        if _signature(written, written.directory.bindings[name]) != \
                _signature(expected, expected.directory.bindings[name]):
            raise SystemExit(f"verification failed for binding {name!r}")
    print(f"WROTE {args.out} ({written.payload_bytes:,} payload bytes, verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
