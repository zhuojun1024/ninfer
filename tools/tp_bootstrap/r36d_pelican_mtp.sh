#!/bin/bash
# MTP K=2: the user's pelican request with reasoning disabled.
set -u
LOG=/home/zhuojun/prof/serve_supervised.log
cd /home/zhuojun/prof
for i in $(seq 1 120); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  grep -qE 'FATAL|terminate called' "$LOG" 2>/dev/null && break
  sleep 3
done
grep 'listening on' "$LOG" | tail -1
timeout 900 curl -s --max-time 850 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/pelican_none_req.json -o /home/zhuojun/prof/pelican_none_mtp.json \
  -w 'pelican_none_mtp http=%{http_code} wall=%{time_total}s\n' || echo 'curl timeout/failed'
python3 - <<'PYEOF'
import json, collections
try:
    d = json.load(open("pelican_none_mtp.json"))
except Exception as exc:
    print("no response:", exc); raise SystemExit(0)
m = d["choices"][0]["message"]
r = m.get("reasoning_content") or ""; c = m.get("content") or ""
u = d.get("usage", {}); t = d.get("timings", {})
print("MTP tokens=%s finish=%s reason=%d content=%d %.1f tok/s" % (
    u.get("completion_tokens"), d["choices"][0].get("finish_reason"), len(r), len(c),
    t.get("predicted_per_second", -1)))
print("content head:", repr(c[:180]))
text = c
w = collections.Counter(text[i:i+60] for i in range(0, max(0, len(text)-60), 3))
print("top repeated 60-char windows:", [(x[:50], n) for x, n in w.most_common(2)])
PYEOF
