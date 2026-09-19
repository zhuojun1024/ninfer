#!/bin/bash
# Compare the two A/B captures token-for-token.
set -u
python3 - <<'PY'
import json
def load(tag):
    return [json.loads(l) for l in open("/home/zhuojun/prof/ab-%s.jsonl" % tag, encoding="utf-8")]
a = load("control"); b = load("split")
ok = True
for x, y in zip(a, b):
    same = x.get("hash") == y.get("hash") and x.get("len") == y.get("len") and x.get("text") == y.get("text")
    ok = ok and same
    print("req%d identical=%s hash=%s len=%d" % (x["i"], same, x.get("hash"), x["len"]))
print("ALL_IDENTICAL" if ok else "MISMATCH")
PY
