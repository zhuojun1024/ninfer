import json, pathlib, sys
sys.path.insert(0, "/home/zhuojun/ninfer")
from tools.artifact.reader import Artifact
p = pathlib.Path("/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer")
with Artifact(p) as a:
    d = a.directory
    for name in ["text/output_head", "text/token_embedding", "text/layers/0/gdn/convolution",
                 "text/layers/0/gdn/query", "text/layers/0/attention/query",
                 "text/layers/0/mlp/gate_up", "mtp/output_head", "proposal/head", "proposal/token_ids",
                 "vision/…"]:
        if name in d.bindings:
            print(name, "->", json.dumps(d.bindings[name])[:400])
    ks = collections_keys = {}
    import collections
    keycount = collections.Counter()
    for name, b in d.bindings.items():
        keycount[tuple(sorted(b.keys()))] += 1
    print("keysets:", keycount.most_common(10))
