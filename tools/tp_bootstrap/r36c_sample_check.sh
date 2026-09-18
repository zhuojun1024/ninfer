#!/bin/bash
# Verify the sampled (server-default) MTP route: two requests without sampling fields, plus timings.
set -u
cd /home/zhuojun/prof
BODY='{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a detailed paragraph about water cycle."}],"max_tokens":128}'
for tag in a b; do
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$BODY" -o "samp_${tag}.json" -w "samp_${tag} http=%{http_code} wall=%{time_total}s\n"
done
python3 - <<'PYEOF'
import json
for tag in ("a", "b"):
    d = json.load(open("samp_%s.json" % tag))
    m = d["choices"][0]["message"]
    t = d.get("timings", {}); u = d.get("usage", {})
    print("--- samp_%s completion=%s predicted_n=%s -> %.2f tok/s" % (
        tag, u.get("completion_tokens"), t.get("predicted_n"), t.get("predicted_per_second", -1)))
    print("reason[:80]:", repr((m.get("reasoning_content") or "")[:80]))
    print("content[:160]:", repr((m.get("content") or "")[:160]))
PYEOF
grep -icE 'error|fatal' serve_supervised.log || true
tail -3 serve_supervised.log
