#!/usr/bin/env python3
"""Drop the orphaned probe comment paragraphs left after the probe blocks were removed."""
import pathlib

PATH = pathlib.Path("/mnt/d/Documents/workbench/ninfer/src/runtime/engine/tp2_generation_core.cpp")
lines = PATH.read_text().splitlines(keepends=True)
keep = []
i = 0
dropped = 0
while i < len(lines):
    if lines[i].lstrip().startswith("//") and "NINFER_TP2_" in lines[i]:
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("//"):
            j += 1
        dropped += j - i
        i = j
        continue
    keep.append(lines[i])
    i += 1
PATH.write_text("".join(keep))
print("dropped comment lines: %d" % dropped)
print("remaining NINFER_TP2 refs:", sum(1 for l in keep if "NINFER_TP2_" in l))
