import sys, json
sys.path.insert(0, "/mnt/d/Documents/workbench/ninfer/tools")
from artifact.reader import Artifact
from artifact.schema import TensorObject
with Artifact("/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer") as art:
    d = art.directory
    obj_shape = {}
    for obj in art.objects:
        if isinstance(obj, TensorObject):
            obj_shape[obj.id] = tuple(obj.shape)
    want = ["text/layers/0/attention/query","text/layers/0/attention/key",
            "text/layers/0/attention/gate","text/layers/0/attention/value",
            "text/layers/0/gdn/query","text/layers/0/gdn/key",
            "text/layers/0/gdn/value","text/layers/0/gdn/z",
            "text/layers/0/gdn/a_projection","text/layers/0/gdn/b_projection",
            "text/layers/0/gdn/convolution","text/layers/0/attention/output",
            "text/layers/0/gdn/output"]
    for name in want:
        b = d.bindings.get(name)
        if b is None:
            print(name + " <no binding>"); continue
        if "object" in b:
            oid = b["object"]; shp = obj_shape.get(oid)
            print(name, "WHOLE", oid, shp)
        else:
            for p in b["parts"]:
                oid = p["object"]; lo,hi = p["range"]; shp = obj_shape.get(oid)
                k = shp[1] if shp and len(shp)>=2 else 5120
                print(name, "part", oid, shp, "rows[%d,%d)" % (lo//k, hi//k))