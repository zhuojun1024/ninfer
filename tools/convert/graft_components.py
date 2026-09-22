"""Combine stored components of one v3 artifact into another, byte for byte.

A v3 binding names the packed object that holds its tensor and the element interval it views
inside it. Object format and byte length follow from the recipe rather than from the conversion
run, so a component's stored bytes can be transplanted without dequantizing, re-quantizing or
re-laying out the payload. That builds a comparison artifact cheaply when the interesting
difference lives in one component (a draft converted from another source checkpoint, for example)
and rebuilding the whole model from source weights would take an hour.

Only packed objects whose every view belongs to a grafted component are copied, so a shared object
can never leak foreign bytes into the result.

Usage:
  python tools/convert/graft_components.py --base USER.ninfer --donor OFFICIAL.ninfer \
      --components vision,mtp,dflash2 --name w4a4-official-parts --out MIXED.ninfer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

MAGIC = b"NINFER\x00\x03"
HEADER_BYTES = 32
CHUNK = 16 * 1024 * 1024


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class Entry:
    """One single-file artifact entry: header, decoded directory and payload placement."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file_bytes = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(HEADER_BYTES)
            if len(header) != HEADER_BYTES or header[:8] != MAGIC:
                raise SystemExit(f"{path}: not a v3 entry (magic {header[:8]!r})")
            self.json_bytes = struct.unpack_from("<Q", header, 8)[0]
            self.artifact_id = header[16:32]
            raw = handle.read(self.json_bytes)
            if len(raw) != self.json_bytes:
                raise SystemExit(f"{path}: truncated directory")
        self.directory = json.loads(raw.decode("utf-8"))
        self.payload_start = align_up(HEADER_BYTES + self.json_bytes, 4096)
        files = self.directory["files"]
        if len(files) != 1:
            raise SystemExit(f"{path}: grafting requires a single-file artifact, found {len(files)}")
        if int(files[0]["payload_bytes"]) != self.file_bytes - self.payload_start:
            raise SystemExit(f"{path}: payload table disagrees with the file length")
        self.objects = {str(o["id"]): o for o in self.directory["objects"]}
        self.bindings = self.directory["bindings"]
        self.binding_objects = {}
        for name, value in self.bindings.items():
            ids = self.referenced_objects(value)
            if ids:
                self.binding_objects[name] = ids

    @staticmethod
    def referenced_objects(value) -> list:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict) and "parts" in value:
            return [str(part["object"]) for part in value["parts"]]
        if isinstance(value, dict):
            for key in ("object", "id", "target"):
                if key in value:
                    return [str(value[key])]
        return []


def plan_graft(base: Entry, donor: Entry, components: list) -> dict:
    users = {}
    for name, ids in base.binding_objects.items():
        for object_id in ids:
            users.setdefault(object_id, set()).add(name)

    objects = {}
    counts = {}
    for component in components:
        names = sorted(n for n in base.binding_objects if n.startswith(component + "/"))
        donor_names = sorted(n for n in donor.binding_objects if n.startswith(component + "/"))
        if names != donor_names:
            raise SystemExit(f"{component}: binding sets differ "
                             f"(base {len(names)}, donor {len(donor_names)})")
        if not names:
            raise SystemExit(f"{component}: nothing to graft")
        for name in names:
            target, source = base.binding_objects[name], donor.binding_objects[name]
            if target != source:
                raise SystemExit(f"{component}: {name} views different objects "
                                 f"(base {target}, donor {source})")
            for object_id in target:
                foreign = sorted(u for u in users.get(object_id, ())
                                 if u.split("/")[0] not in components)
                if foreign:
                    raise SystemExit(f"{object_id} is also a view of {foreign[:3]}; "
                                     f"refusing an ambiguous graft")
                objects[object_id] = object_id
        counts[component] = len(names)

    for object_id in objects:
        target, source = base.objects[object_id], donor.objects.get(object_id)
        if source is None:
            raise SystemExit(f"{object_id} is missing from the donor")
        if int(target["bytes"]) != int(source["bytes"]) or target.get("format") != source.get("format"):
            raise SystemExit(f"{object_id} differs (base {target.get('format')}/{target['bytes']}, "
                             f"donor {source.get('format')}/{source['bytes']})")
    return objects, counts


def short_hash(handle, offset: int, length: int) -> str:
    handle.seek(offset)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        block = handle.read(min(CHUNK, remaining))
        if not block:
            raise SystemExit("unexpected end of file while hashing")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()[:16]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="artifact that keeps its payload")
    parser.add_argument("--donor", type=Path, required=True, help="artifact read for components")
    parser.add_argument("--components", default="vision,mtp,dflash2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None, help="value stored in metadata.name")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    components = [c.strip() for c in args.components.split(",") if c.strip()]
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to replace it")
    if args.out.resolve() in (args.base.resolve(), args.donor.resolve()):
        raise SystemExit("--out must be a new path")

    base = Entry(args.base)
    donor = Entry(args.donor)
    objects, counts = plan_graft(base, donor, components)

    total = sum(int(base.objects[o]["bytes"]) for o in objects)
    print("transplant plan")
    for component, count in counts.items():
        print(f"  {component}: {count} bindings")
    by_format = {}
    for object_id in objects:
        key = str(base.objects[object_id].get("format"))
        by_format[key] = by_format.get(key, 0) + 1
    print(f"  {len(objects)} objects, {total / 2**20:.1f} MiB, formats " +
          ", ".join(f"{k} x{v}" for k, v in sorted(by_format.items())))

    shutil.copyfile(base.path, args.out)
    with args.out.open("r+b") as out, donor.path.open("rb") as src:
        metadata = base.directory["metadata"]
        metadata["graft"] = {
            "donor": donor.path.name,
            "donor_artifact_id": donor.artifact_id.hex(),
            "components": components,
            "objects": len(objects),
            "bytes": total,
        }
        if args.name:
            metadata["name"] = args.name
        text = json.dumps(base.directory, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(text) > base.json_bytes:
            raise SystemExit(f"patched directory needs {len(text)} bytes, entry has {base.json_bytes}")
        out.seek(HEADER_BYTES)
        out.write(text + b" " * (base.json_bytes - len(text)))
        out.seek(16)
        out.write(os.urandom(16))

        copied = 0
        for object_id in sorted(objects):
            size = int(base.objects[object_id]["bytes"])
            out.seek(base.payload_start + int(base.objects[object_id]["offset"]))
            src.seek(donor.payload_start + int(donor.objects[object_id]["offset"]))
            remaining = size
            while remaining:
                block = src.read(min(CHUNK, remaining))
                if not block:
                    raise SystemExit(f"{donor.path}: truncated payload")
                out.write(block)
                remaining -= len(block)
            copied += size

        print(f"copied {copied / 2**20:.1f} MiB into {args.out}")
        mismatches = []
        for object_id in sorted(objects):
            size = int(base.objects[object_id]["bytes"])
            a = short_hash(out, base.payload_start + int(base.objects[object_id]["offset"]), size)
            b = short_hash(src, donor.payload_start + int(donor.objects[object_id]["offset"]), size)
            if a != b:
                mismatches.append(object_id)
        if mismatches:
            raise SystemExit(f"verification failed for {len(mismatches)} objects: {mismatches[:3]}")
        print("verified: every transplanted object matches the donor byte for byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
