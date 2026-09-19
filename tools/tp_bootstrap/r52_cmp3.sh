#!/bin/bash
set -u
python3 - <<'PY'
import json
def load(tag):
    return [json.loads(l) for l in open("/home/zhuojun/prof/ab-%s.jsonl" % tag, encoding="utf-8")]
ref = load("embed")
for tag in ("cur131", "ctx262", "ctx262b"):
    cur = load(tag)
    ok = all(x.get("hash") == y.get("hash") for x, y in zip(ref, cur))
    print(tag, "IDENTICAL" if ok else "DIFFERS")
    if not ok:
        for x, y in zip(ref, cur):
            if x.get("hash") != y.get("hash"):
                print("   req%d ref %s(%d) vs %s(%d)" % (x["i"], x["hash"], x["len"], y["hash"], y["len"]))
PY
