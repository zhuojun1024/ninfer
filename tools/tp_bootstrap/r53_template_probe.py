import hashlib, json, pathlib, sys
sys.path.insert(0, "/home/zhuojun/ninfer")
from tools.artifact.reader import Artifact

local = pathlib.Path("/home/zhuojun/models/chat_template.jinja").read_bytes()
print("local chat_template.jinja sha256", hashlib.sha256(local).hexdigest(), len(local), "bytes")

with Artifact(pathlib.Path("/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer")) as a:
    ids = [o.id for o in a.objects]
    hits = [i for i in ids if "chat_template" in i or "tokenizer_config" in i or "generation_config" in i]
    print("resource objects:", hits)
    for name in hits:
        blob = a.read_object(name)
        print(" ", name, len(blob), "sha256", hashlib.sha256(blob).hexdigest()[:16])
        if name.endswith("chat_template.jinja"):
            print("   identical to the local file:", blob == local)
        if name.endswith("tokenizer_config.json"):
            cfg = json.loads(blob)
            inner = cfg.get("chat_template")
            if isinstance(inner, str):
                print("   tokenizer_config.chat_template sha256",
                      hashlib.sha256(inner.encode()).hexdigest()[:16])
