#!/bin/bash
# Loop probe: two prompts on whatever route is currently serving, tagged for A/B.
set -u
TAG="${1:?tag}"
cd /home/zhuojun/prof
P1='{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a detailed paragraph about water cycle."}],"max_tokens":900}'
P2='{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a very long, extremely detailed step-by-step reasoning (at least 1500 words) proving that the sum of the first n odd numbers equals n squared, covering every intermediate detail and every edge case."}],"max_tokens":1500}'
i=1
for BODY in "$P1" "$P2"; do
  curl -s --max-time 900 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$BODY" -o "${TAG}_p${i}.json" -w "${TAG}_p${i} http=%{http_code} wall=%{time_total}s\n"
  i=$((i+1))
done
python3 - "$TAG" <<'PYEOF'
import json, sys, collections
tag = sys.argv[1]
for i in (1, 2):
    d = json.load(open("%s_p%d.json" % (tag, i)))
    m = d["choices"][0]["message"]
    t = d.get("timings", {}); u = d.get("usage", {})
    reason = m.get("reasoning_content") or ""
    content = m.get("content") or ""
    text = reason + "||" + content
    w = collections.Counter(text[j:j+40] for j in range(0, max(0, len(text) - 40), 5))
    top = w.most_common(1)
    print("%s_p%d tokens=%s %.1f tok/s finish=%s reason=%d content=%d" % (
        tag, i, u.get("completion_tokens"), t.get("predicted_per_second", -1),
        d["choices"][0].get("finish_reason"), len(reason), len(content)))
    if top:
        print("   most repeated 40-char window x%d: %r" % (top[0][1], top[0][0].replace(chr(10), " ")))
    print("   reason tail 160: %r" % reason[-160:])
PYEOF
