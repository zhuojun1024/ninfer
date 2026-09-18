#!/bin/bash
LOG=/home/zhuojun/prof/serve_supervised.log
for i in $(seq 1 120); do
  if tail -40 "$LOG" | grep -q 'listening on'; then break; fi
  if tail -20 "$LOG" | grep -qE 'error|Error|terminate|invalid'; then break; fi
  sleep 2
done
echo '--- last lines ---'; tail -6 "$LOG"
echo '--- procs ---'; pgrep -af 'ninfer-serve'
echo '--- request ---'
curl -s --max-time 180 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"name three colors"}],"max_tokens":6}' \
  -o /home/zhuojun/prof/mtp_smoke.json -w 'http=%{http_code} wall=%{time_total}s\n'
grep -E 'req#1 done' "$LOG" | tail -1
