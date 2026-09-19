import collections, json, pathlib, sys
sys.path.insert(0, "/home/zhuojun/ninfer")
from tools.artifact.reader import Artifact
p = pathlib.Path("/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer")
with Artifact(p) as a:
    d = a.directory
    print("file_bytes", a.file_bytes, "payload_bytes", a.payload_bytes)
    print("components", json.dumps(d.components)[:1200] if not isinstance(d.components, dict) else json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("config",)}) for k, v in d.components.items()})[:2500])
    objs = a.objects
    print("objects", len(objs))
    ids = [getattr(o, "id", None) for o in objs[:8]]
    print("sample ids", ids)
