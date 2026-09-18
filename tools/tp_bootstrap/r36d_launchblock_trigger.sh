#!/bin/bash
# Fire the crashing request at the launch-blocking serve and report where the fault surfaces.
set -u
LOG=/home/zhuojun/prof/launchblock.log
for i in $(seq 1 240); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  grep -qE 'FATAL|LAUNCHBLOCK_SERVE_EXIT' "$LOG" 2>/dev/null && break
  sleep 3
done
grep 'listening on' "$LOG" | tail -1
for attempt in 1 2; do
  curl -s --max-time 1800 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a detailed paragraph about water cycle."}],"max_tokens":900}' \
    -o /home/zhuojun/prof/lb_repro.json -w "attempt$attempt http=%{http_code} wall=%{time_total}s\n"
  pgrep -f 'build/apps/ninfer-serve' > /dev/null || { echo "server gone"; break; }
  python3 -c "import json; d=json.load(open('/home/zhuojun/prof/lb_repro.json')); u=d.get('usage',{}); print('   ok tokens=%s finish=%s' % (u.get('completion_tokens'), d['choices'][0].get('finish_reason')));" || true
done
echo '--- error lines ---'
grep -nE 'failed|error|Error|EXIT' "$LOG" | tail -12
echo '--- last rounds ---'
grep '\[mtp\] round' "$LOG" | tail -2
