#!/usr/bin/env python3
"""Remove the temporary TP-2 debug probes (env-guarded blocks) from the generation core."""
import pathlib

PATH = pathlib.Path("/mnt/d/Documents/workbench/ninfer/src/runtime/engine/tp2_generation_core.cpp")
lines = PATH.read_text().splitlines(keepends=True)


def block_end(start):
    depth = 0
    for idx in range(start, len(lines)):
        depth += lines[idx].count("{") - lines[idx].count("}")
        if depth == 0 and idx > start:
            return idx
    raise RuntimeError("unbalanced block at line %d" % (start + 1))


out = []
i = 0
removed = []
while i < len(lines):
    line = lines[i]
    if 'NINFER_TP2_SKIP_FOLD' in line:
        end = block_end(i)
        body = lines[i + 1:end]
        out.extend([b[4:] if b.startswith("    ") else b for b in body])
        removed.append("skip-fold@%d" % (i + 1))
        i = end + 1
        continue
    if 'if (std::getenv("NINFER_TP2_' in line:
        start = i
        while start > 0 and lines[start - 1].lstrip().startswith("//"):
            start -= 1
        end = block_end(i)
        nxt = end + 1
        if nxt < len(lines) and lines[nxt].strip() == "":
            nxt += 1
        removed.append("probe@%d" % (start + 1))
        i = nxt
        continue
    out.append(line)
    i += 1

PATH.write_text("".join(out))
print("removed:", ", ".join(removed))
print("remaining NINFER_TP2 refs:", sum(1 for l in out if "NINFER_TP2_" in l))
