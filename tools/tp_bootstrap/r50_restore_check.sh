#!/bin/bash
LOG=/home/zhuojun/prof/r50_restore.log
: > "$LOG"
for i in $(seq 1 80); do
  pgrep -a -f 'ninfer-serv[e]' | grep -q 'lm-head-draft' || { sleep 3; continue; }
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8088/health)
  [ "$code" = "200" ] && break
  sleep 3
done
echo "--- process ---" >> "$LOG"
pgrep -a -f 'ninfer-serv[e]' >> "$LOG" 2>&1
echo "--- memory ---" >> "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >> "$LOG" 2>&1
curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/live_req.json -o /home/zhuojun/prof/r50_live.json \
  -w "live http=%{http_code} wall=%{time_total}s\n" >> "$LOG" 2>&1
python3 - >> "$LOG" 2>&1 <<'PY'
import json
d = json.load(open("/home/zhuojun/prof/r50_live.json"))
ch = d["choices"][0]; m = ch["message"]
c = m.get("content") or ""; r = m.get("reasoning_content") or ""
t = d.get("timings", {}) or {}
print("finish=%s content=%d reason=%d tok_s=%.2f" % (ch.get("finish_reason"), len(c), len(r), t.get("predicted_per_second", 0.0)))
print("CONTENT:", (c or r)[:300].replace("\n", " "))
PY
echo "RESTORE_CHECK_DONE" >> "$LOG"
cat "$LOG"
