#!/bin/bash
# Pelican prompt on the plain route: same sampling defaults, same output limit.
set -u
LOG=/home/zhuojun/prof/plain_pelican.log
cd /home/zhuojun/prof
for i in $(seq 1 120); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  grep -qE 'FATAL|error' "$LOG" 2>/dev/null && break
  sleep 3
done
grep 'listening on' "$LOG" | tail -1
timeout 600 curl -s --max-time 550 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/pelican_req.json -o /home/zhuojun/prof/pelican_plain.json \
  -w 'pelican_plain http=%{http_code} wall=%{time_total}s\n' || echo 'curl timeout/failed'
python3 - <<'PYEOF'
import json, collections
try:
    d = json.load(open("pelican_plain.json"))
except Exception as exc:
    print("no response:", exc); raise SystemExit(0)
m = d["choices"][0]["message"]
r = m.get("reasoning_content") or ""; c = m.get("content") or ""
u = d.get("usage", {}); t = d.get("timings", {})
print("tokens=%s finish=%s reason=%d content=%d %.1f tok/s" % (
    u.get("completion_tokens"), d["choices"][0].get("finish_reason"), len(r), len(c),
    t.get("predicted_per_second", -1)))
text = r + "||" + c
w = collections.Counter(text[i:i+60] for i in range(0, max(0, len(text)-60), 3))
for win, n in w.most_common(2):
    print("  x%d %r" % (n, win.replace(chr(10), " ")[:60]))
print("  reason tail:", repr(r[-160:]))
print("  content head:", repr(c[:160]))
PYEOF
