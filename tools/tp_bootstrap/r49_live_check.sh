#!/bin/bash
# Live sanity check of the 8088 service (fp8 KV 131072 + MTP K=2).
set -u
python3 - <<'PY'
import json
body = {"model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "用一句话说明什么是张量并行。"}],
        "max_tokens": 200}
open("/home/zhuojun/prof/live_req.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
PY
curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/live_req.json -o /home/zhuojun/prof/live1.json \
  -w "live http=%{http_code} wall=%{time_total}s\n"
python3 - <<'PY'
import json, re
d = json.load(open("/home/zhuojun/prof/live1.json"))
ch = d["choices"][0]; m = ch["message"]
c = m.get("content") or ""; r = m.get("reasoning_content") or ""
print("finish=%s content=%d reason=%d" % (ch.get("finish_reason"), len(c), len(r)))
print("CONTENT:", (c or r)[:400].replace("\n", " "))
PY
