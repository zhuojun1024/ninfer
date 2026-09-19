import json, pathlib, sys
sys.path.insert(0, "/home/zhuojun/ninfer")
from tools.artifact.reader import Artifact
p = pathlib.Path("/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer")
with Artifact(p) as a:
    d = a.directory
    print("bindings type", type(d.bindings))
    names = list(d.bindings.keys())
    print("n bindings", len(names))
    print("names sample", names[:12])
    b = d.bindings["text/output_head"]
    print("binding repr", repr(b)[:800])
    print("binding type", type(b))
    objs = a.objects
    print("obj0", repr(objs[0])[:300])
