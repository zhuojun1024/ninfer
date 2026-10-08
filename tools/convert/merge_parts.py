"""Merge selected components (mtp/dflash2/vision) from a donor into a text artifact.

Two subcommands:

  extract  Take the selected components out of an artifact as a small reusable
           parts artifact (e.g. 2.6 GB instead of 23 GB), for repeated merging.

  merge    Combine the text component of a base artifact (verbatim, including
           its own object ids, proposal and resources) with the selected
           components of a parts artifact into one new artifact. The base's
           mtp/dflash2/vision components, when present, are replaced by the
           donor's; a donor component absent from the base is added.

           When the donor's text component declares a proposal table
           (text.proposal, the reduced lm-head that --lm-head-draft reads) and
           the base has none, the proposal sub-tree is transplanted too: the
           text.proposal field, the proposal/* bindings and their objects, and
           the Uses that reference a proposal parameter. This requires the
           donor's and the base's text config to match, since the proposal
           head is a reduced view of the same lm-head. A parts artifact made by
           extract never carries a proposal, so this only kicks in when merging
           with a full donor that has one.

The text side never crosses the merge boundary: components are self-contained
(no object is shared across components), so the merge is a byte-exact
transplant with a compact relayout. Object ids that collide between the two
sides are renamed on the donor side and their binding/use/resource references
are rewritten. Output bytes are verified against the sources after writing.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.reader import Artifact
from tools.artifact.schema import (
    ArtifactError,
    ResourceObject,
    ResourceSpec,
    TensorObject,
    TensorSpec,
)
from tools.artifact.writer import ArtifactWriter

DEFAULT_COMPONENTS = "mtp,dflash2,vision"
HASH_CHUNK = 64 * 1024 * 1024


def component_of(parameter: str) -> str:
    return parameter.split("/", 1)[0]


def component_object_ids(directory, component: str) -> set[str]:
    """Every object a component references: bindings, resources, use auxiliaries."""
    ids: set[str] = set()
    prefix = f"{component}/"
    for name, binding in directory.bindings.items():
        if not name.startswith(prefix):
            continue
        if "object" in binding:
            ids.add(binding["object"])
        else:
            ids.update(part["object"] for part in binding["parts"])
    component_record = directory.components.get(component, {})
    ids.update(component_record.get("resources", {}).values())
    for use in directory.uses:
        if not use["parameter"].startswith(prefix):
            continue
        for auxiliary in use.get("auxiliaries", {}).values():
            if "object" in auxiliary:
                ids.add(auxiliary["object"])
            else:
                ids.update(part["object"] for part in auxiliary["parts"])
    return ids


def spec_of(obj: TensorObject | ResourceObject):
    if isinstance(obj, ResourceObject):
        return ResourceSpec(obj.id, obj.bytes, obj.encoding)
    return TensorSpec(obj.id, obj.shape, obj.format, obj.layout)


def remap_binding(binding: dict, idmap: dict[str, str]) -> dict:
    if "object" in binding:
        return {"object": idmap.get(binding["object"], binding["object"])}
    return {
        "parts": [
            {
                "object": idmap.get(part["object"], part["object"]),
                "range": part["range"],
            }
            for part in binding["parts"]
        ]
    }


def remap_use(use: dict, idmap: dict[str, str]) -> dict:
    result = {key: value for key, value in use.items() if key != "auxiliaries"}
    auxiliaries = use.get("auxiliaries")
    if auxiliaries:
        result["auxiliaries"] = {
            role: remap_binding(auxiliary, idmap)
            for role, auxiliary in auxiliaries.items()
        }
    return result


def remap_component(component: dict, idmap: dict[str, str]) -> dict:
    result = dict(component)
    resources = component.get("resources")
    if resources:
        result["resources"] = {
            role: idmap.get(obj_id, obj_id) for role, obj_id in resources.items()
        }
    return result


def parse_components(value: str) -> list[str]:
    components = [item.strip() for item in value.split(",") if item.strip()]
    if not components or any(component in ("text", "") for component in components):
        raise ArtifactError(
            f"components must be a nonempty list from mtp/dflash2/vision, got {value!r}"
        )
    if len(set(components)) != len(components):
        raise ArtifactError(f"duplicate components in {value!r}")
    return components


def require_single_file(artifact: Artifact, label: str) -> None:
    if len(artifact.directory.files) != 1:
        raise ArtifactError(f"{label} must be a single-file artifact")


def sha256_chunks(source) -> str:
    digest = hashlib.sha256()
    for chunk in source:
        digest.update(chunk)
    return digest.hexdigest()


def stream_object(writer: ArtifactWriter, artifact: Artifact, obj_id: str, out_id: str) -> None:
    writer.write_object(out_id, artifact.iter_object(obj_id))


def verify_output(
    path: str,
    pairs: list[tuple[str, Artifact, str]],
) -> None:
    """Hash-compare every output object region against its source region."""
    with Artifact.open(path) as output:
        for out_id, source, src_id in pairs:
            expected = sha256_chunks(source.iter_object(src_id))
            actual = sha256_chunks(output.iter_object(out_id))
            if expected != actual:
                raise ArtifactError(
                    f"{out_id}: content hash differs from source {src_id}"
                )


def source_order(directory, ids: set[str]) -> list[str]:
    return [obj.id for obj in directory.objects if obj.id in ids]


def cmd_extract(args: argparse.Namespace) -> int:
    components = parse_components(args.components)
    with Artifact.open(args.source) as donor:
        require_single_file(donor, "--source")
        missing = [c for c in components if c not in donor.directory.components]
        if missing:
            raise ArtifactError(
                f"source has no component {missing}; has "
                f"{sorted(donor.directory.components)}"
            )
        selected_ids = set()
        for component in components:
            ids = component_object_ids(donor.directory, component)
            if not ids:
                raise ArtifactError(f"component {component} references no objects")
            selected_ids |= ids
        object_ids = source_order(donor.directory, selected_ids)
        specs = [spec_of(donor.by_id[obj_id]) for obj_id in object_ids]
        bindings = {
            name: binding
            for name, binding in donor.directory.bindings.items()
            if component_of(name) in components
        }
        uses = [
            use
            for use in donor.directory.uses
            if component_of(use["parameter"]) in components
        ]
        out_components = {
            "text": {"config": donor.directory.components["text"]["config"]}
        }
        out_components.update(
            {c: donor.directory.components[c] for c in components}
        )
        name = args.name or (
            f"{donor.directory.metadata.get('name', donor.path.stem)}-parts"
        )
        provenance = {
            "extract": {
                "source": str(donor.path),
                "components": components,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        }
        writer = ArtifactWriter(
            args.out,
            specs,
            components=out_components,
            bindings=bindings,
            uses=uses,
            metadata={"name": name},
            provenance=provenance,
        )
        try:
            print(f"extracting {len(specs)} objects from {donor.path}")
            for index, spec in enumerate(specs, start=1):
                print(
                    f"[{index}/{len(specs)}] {spec.id}", end=" ", flush=True
                )
                stream_object(writer, donor, spec.id, spec.id)
                print("ok")
            directory = writer.finish()
        except BaseException:
            writer.abort()
            raise
        if not args.no_verify:
            print("verifying output against source ...")
            verify_output(str(writer.path), [(s.id, donor, s.id) for s in specs])
            print("verification passed")
    print(
        f"wrote {writer.path} ({writer.path.stat().st_size} bytes, "
        f"{len(directory.objects)} objects, {len(bindings)} bindings, {len(uses)} uses)"
    )
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    components = parse_components(args.components)
    out = args.out or str(__import__("pathlib").Path(args.base).parent / f"{args.name}.ninfer")
    with Artifact.open(args.base) as base, Artifact.open(args.parts) as parts:
        require_single_file(base, "--base")
        require_single_file(parts, "--parts")
        selected = [c for c in components if c in parts.directory.components]
        if not selected:
            raise ArtifactError(
                f"parts artifact has none of the requested components {components}; "
                f"has {sorted(parts.directory.components)}"
            )
        # The text component's proposal sub-tree: when the donor's text declares a
        # proposal table and the base has none, transplant it so a single merge can
        # produce a --lm-head-draft ready artifact. "proposal" is treated as a
        # pseudo-component so the existing binding/use/object/rename machinery picks
        # it up; only the text.proposal field must be written separately.
        donor_proposal = parts.directory.components.get("text", {}).get("proposal")
        base_proposal = base.directory.components.get("text", {}).get("proposal")
        transplant_proposal = False
        if donor_proposal is not None:
            if base_proposal is None:
                if parts.directory.components["text"]["config"] != \
                        base.directory.components["text"]["config"]:
                    raise ArtifactError(
                        "donor carries a proposal table but its text config differs "
                        "from the base's; the proposal head cannot be shared"
                    )
                transplant_proposal = True
            elif base_proposal != donor_proposal:
                raise ArtifactError(
                    "base and parts declare different proposal tables; merge one "
                    "side's text instead"
                )
        if transplant_proposal:
            selected = selected + ["proposal"]
            print("transplanting donor proposal table (text.proposal + proposal/* objects)")
        dropped = set(selected) & set(base.directory.components)
        dropped_ids = set()
        for component in dropped:
            dropped_ids |= component_object_ids(base.directory, component)
        kept_refs = set()
        for component in (*base.directory.components, "proposal"):
            if component in dropped:
                continue
            kept_refs |= component_object_ids(base.directory, component)
        shared = dropped_ids & kept_refs
        if shared:
            raise ArtifactError(
                f"objects shared between kept and dropped components: {sorted(shared)}"
            )
        unreferenced = (
            {obj.id for obj in base.directory.objects} - kept_refs - dropped_ids
        )
        if unreferenced:
            raise ArtifactError(
                f"base has objects referenced by no component: {sorted(unreferenced)}"
            )

        selected_ids = set()
        for component in selected:
            selected_ids |= component_object_ids(parts.directory, component)
        collision = selected_ids & kept_refs
        idmap: dict[str, str] = {}
        for index, obj_id in enumerate(
            source_order(parts.directory, collision), start=1
        ):
            idmap[obj_id] = f"merged/{index:06d}"
        if idmap:
            print(
                f"renaming {len(idmap)} donor object ids that collide with the base "
                f"(e.g. {next(iter(idmap))} -> {idmap[next(iter(idmap))]})"
            )

        kept_specs = [spec_of(obj) for obj in base.directory.objects if obj.id in kept_refs]
        part_entries: list[tuple, str] = [
            (spec_of(parts.by_id[obj_id]).__replace__(id=idmap.get(obj_id, obj_id)), obj_id)
            for obj_id in source_order(parts.directory, selected_ids)
        ]
        part_specs = [spec for spec, _ in part_entries]
        part_bindings = {
            name: remap_binding(binding, idmap)
            for name, binding in parts.directory.bindings.items()
            if component_of(name) in selected
        }
        part_uses = [
            remap_use(use, idmap)
            for use in parts.directory.uses
            if component_of(use["parameter"]) in selected
        ]
        out_components = {"text": base.directory.components["text"]}
        if transplant_proposal:
            text = dict(base.directory.components["text"])
            text["proposal"] = dict(donor_proposal)
            out_components["text"] = text
        out_components.update(
            {c: remap_component(parts.directory.components[c], idmap)
             for c in selected if c in parts.directory.components}
        )
        out_bindings = {
            name: binding
            for name, binding in base.directory.bindings.items()
            if component_of(name) not in dropped
        }
        out_bindings.update(part_bindings)
        out_uses = [
            use
            for use in base.directory.uses
            if component_of(use["parameter"]) not in dropped
        ]
        out_uses.extend(part_uses)
        provenance = {
            "merge": {
                "base": str(base.path),
                "parts": str(parts.path),
                "components": selected,
                "proposal_transplanted": transplant_proposal,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        }
        writer = ArtifactWriter(
            out,
            kept_specs + part_specs,
            components=out_components,
            bindings=out_bindings,
            uses=out_uses,
            metadata={"name": args.name},
            provenance=provenance,
        )
        try:
            print(f"merging {len(kept_specs)} base objects with {len(part_specs)} donor objects")
            for index, spec in enumerate(kept_specs, start=1):
                print(f"[{index}/{len(kept_specs) + len(part_specs)}] base {spec.id}", end=" ", flush=True)
                stream_object(writer, base, spec.id, spec.id)
                print("ok")
            total = len(kept_specs) + len(part_specs)
            for index, (spec, orig_id) in enumerate(part_entries, start=len(kept_specs) + 1):
                print(f"[{index}/{total}] donor {orig_id}" + (f" -> {spec.id}" if spec.id != orig_id else ""), end=" ", flush=True)
                stream_object(writer, parts, orig_id, spec.id)
                print("ok")
            directory = writer.finish()
        except BaseException:
            writer.abort()
            raise
        pairs: list[tuple[str, Artifact, str]] = [(s.id, base, s.id) for s in kept_specs]
        pairs += [(spec.id, parts, orig_id) for spec, orig_id in part_entries]
        if not args.no_verify:
            print("verifying output against sources ...")
            verify_output(str(writer.path), pairs)
            print("verification passed")
    print(
        f"wrote {writer.path} ({writer.path.stat().st_size} bytes, "
        f"{len(directory.objects)} objects, {len(out_bindings)} bindings, {len(out_uses)} uses)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("extract", "extract the selected components into a parts artifact"),
        ("merge", "merge a text artifact with a parts artifact into a new artifact"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--components",
            default=DEFAULT_COMPONENTS,
            help=f"comma-separated components (default {DEFAULT_COMPONENTS})",
        )
        p.add_argument("--no-verify", action="store_true", help="skip byte verification")
    extract = sub.choices["extract"]
    extract.add_argument("--source", required=True, help="artifact to extract from")
    extract.add_argument("--out", required=True, help="parts artifact to write")
    extract.add_argument("--name", default=None, help="display name for the parts artifact")
    extract.set_defaults(func=cmd_extract)
    merge = sub.choices["merge"]
    merge.add_argument("--base", required=True, help="text artifact to keep")
    merge.add_argument("--parts", required=True, help="parts or full donor artifact")
    merge.add_argument("--name", required=True, help="display name for the merged artifact")
    merge.add_argument("--out", default=None, help="output path (default <base dir>/<name>.ninfer)")
    merge.set_defaults(func=cmd_merge)
    args = parser.parse_args()
    try:
        return args.func(args)
    except ArtifactError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
