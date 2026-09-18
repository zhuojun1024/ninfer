import sys, json
sys.path.insert(0, "/mnt/d/Documents/workbench/ninfer/tools")
from artifact.reader import Artifact
from artifact.schema import TensorObject
from collections import Counter

path = "/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer"
with Artifact(path) as art:
    combos = Counter()
    for obj in art.objects:
        if isinstance(obj, TensorObject):
            combos[(tuple(obj.shape), obj.format)] += 1
    print("=== unique (shape, format) -> count ===")
    for (shape, fmt), c in sorted(combos.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print("  %5d  %-14s %s" % (c, fmt, shape))
    print("=== layer 0 objects ===")
    for obj in art.objects:
        if isinstance(obj, TensorObject) and obj.id.startswith("text/layers/0/"):
            print("  %-52s %s %s" % (obj.id, tuple(obj.shape), obj.format))
