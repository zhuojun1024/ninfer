#!/bin/bash
# Fire the user's pelican prompt at the sanitized MTP serve and dump the first memcheck error.
set -u
LOG=/home/zhuojun/prof/sanitize.log
cd /home/zhuojun/prof
python3 - <<'PYEOF'
import json
body = {"model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "创建一个HTML，内容是SVG绘制一个鹈鹕骑自行车的2D动画"}],
        "max_tokens": 700}
open("/home/zhuojun/prof/pelican_req.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
print("request written")
PYEOF
for i in $(seq 1 120); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  grep -qE 'FATAL|SANITIZED_SERVE_EXIT' "$LOG" 2>/dev/null && break
  sleep 3
done
grep 'listening on' "$LOG" | tail -1
timeout 900 curl -s --max-time 850 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/pelican_req.json -o /home/zhuojun/prof/pelican_san.json \
  -w 'pelican_san http=%{http_code} wall=%{time_total}s\n' || echo 'curl timeout/failed'
sleep 5
echo "--- rounds logged: $(grep -c 'mtp. round' "$LOG") ---"
echo '--- first Invalid block ---'
grep -n -m1 -B2 -A18 'Invalid' "$LOG" || echo 'no Invalid report'
echo '--- error summary ---'
grep -n -A4 'ERROR SUMMARY' "$LOG" | head -12 || true
