#!/usr/bin/env python3
"""Rewrite the per-device shared-memory opt-in in the quantized prompt attention launches.

cudaFuncSetAttribute is a per-device property of the function, so a plain function-local static
only configures the first device that reaches the launch. A tensor-parallel run launches the same
instantiation from every shard context, and a device that never opted in rejects the >48 KiB launch
with cudaErrorInvalidValue.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path("/mnt/d/Documents/workbench/ninfer/src/ops/softmax_attention/dense/causal_cache")
PATTERN = re.compile(
    r"    static const cudaError_t attr = cudaFuncSetAttribute\(\n"
    r"        ([^\n]+),\n"
    r"        cudaFuncAttributeMaxDynamicSharedMemorySize, ([^\n]+)\);\n"
    r"    CUDA_CHECK\(attr\);"
)
REPLACEMENT = (
    "    // Per-device opt-in: a function-local static would configure only the first shard's device.\n"
    "    int device = 0;\n"
    "    CUDA_CHECK(cudaGetDevice(&device));\n"
    "    static bool attr_done[64] = {};\n"
    "    const int attr_slot = (device >= 0 && device < 64) ? device : 0;\n"
    "    if (!attr_done[attr_slot]) {\n"
    "        CUDA_CHECK(cudaFuncSetAttribute(\n"
    "            \g<1>,\n"
    "            cudaFuncAttributeMaxDynamicSharedMemorySize, \g<2>));\n"
    "        attr_done[attr_slot] = true;\n"
    "    }"
)

changed = []
for path in sorted(ROOT.glob("*.cu")):
    text = path.read_text()
    new, count = PATTERN.subn(REPLACEMENT, text)
    if count:
        path.write_text(new)
        changed.append("%s: %d" % (path.name, count))
print("changed:", "; ".join(changed) if changed else "none")
sys.exit(0 if changed else 1)
