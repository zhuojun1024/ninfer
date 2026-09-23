"""Print draft-related formats and the conversion provenance of a v3 artifact.

Usage: python tools/tp_bootstrap/r54_draft_quant_inspect.py <artifact.ninfer>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.reader import Artifact  # noqa: E402
from tools.artifact.schema import TensorObject  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1])
    with Artifact(path) as artifact:
        directory = artifact.directory
        tensors = [obj for obj in directory.objects if isinstance(obj, TensorObject)]
        index = {obj.id: obj for obj in directory.objects}
        print(json.dumps(
            {
                "path": str(path),
                "name": directory.metadata.get("name"),
                "objects": len(directory.objects),
                "formats": dict(sorted(Counter(obj.format for obj in tensors).items())),
                "provenance": directory.provenance,
            },
            ensure_ascii=False,
            indent=2,
        ))
        print("--- dflash2/mtp/proposal bindings ---")
        for name in sorted(directory.bindings):
            if not name.startswith(("dflash2/", "mtp/", "proposal/")):
                continue
            binding = directory.bindings[name]
            if "object" in binding:
                obj = index[binding["object"]]
                print(f"{name} -> {obj.id}:{obj.format}:{list(obj.shape)}")
                continue
            parts = []
            for part in binding["parts"]:
                obj = index[part["object"]]
                parts.append(f"{part['object']}:{obj.format}:{list(obj.shape)}")
            print(f"{name} -> {', '.join(parts)}")


if __name__ == "__main__":
    main()
