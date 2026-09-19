#!/bin/bash
# Deterministic greedy A/B suite (top_k=1 pins argmax) for the vocabulary-parallel head.
set -u
PROF=/home/zhuojun/prof
TAG="$1"
python3 - "$TAG" <<'PY'
import json, sys
prof = "/home/zhuojun/prof"
tag = sys.argv[1]
prompts = [
    "\u7528\u4e00\u53e5\u8bdd\u89e3\u91ca\u4ec0\u4e48\u662f\u5f20\u91cf\u5e76\u884c\u3002",
    "\u628a\u201c\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u4eec\u53bb\u516c\u56ed\u6563\u6b65\u3002\u201d\u7ffb\u8bd1\u6210\u82f1\u6587\u3002",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "3+4*5 \u7b49\u4e8e\u591a\u5c11\uff1f\u53ea\u56de\u7b54\u6570\u5b57\u3002",
    "List three differences between TCP and UDP, one line each.",
]
for i, p in enumerate(prompts):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": p}],
            "max_tokens": 160, "temperature": 0, "top_k": 1, "top_p": 1.0}
    json.dump(body, open("%s/ab-req-%d.json" % (prof, i), "w", encoding="utf-8"), ensure_ascii=False)
PY
for i in 0 1 2 3 4; do
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @$PROF/ab-req-$i.json -o "$PROF/ab-$TAG-$i.json" \
    -w "req-$i http=%{http_code} wall=%{time_total}s\n"
done
python3 - "$TAG" <<'PY'
import hashlib, json, sys
prof = "/home/zhuojun/prof"
tag = sys.argv[1]
rows = []
for i in range(5):
    d = json.load(open("%s/ab-%s-%d.json" % (prof, tag, i)))
    if "choices" not in d:
        print("req%d ERROR %s" % (i, str(d)[:200])); rows.append({"i": i, "error": str(d)[:200]}); continue
    ch = d["choices"][0]; m = ch["message"]
    text = (m.get("content") or "") + "||" + (m.get("reasoning_content") or "")
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    rows.append({"i": i, "finish": ch.get("finish_reason"), "hash": h, "len": len(text), "text": text})
    print("req%d finish=%s hash=%s len=%d" % (i, ch.get("finish_reason"), h, len(text)))
with open("%s/ab-%s.jsonl" % (prof, tag), "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", "%s/ab-%s.jsonl" % (prof, tag))
PY
