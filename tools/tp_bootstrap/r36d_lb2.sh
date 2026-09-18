#!/bin/bash
# Fire the immediately-crashing request under CUDA_LAUNCH_BLOCKING to localize the failing launch.
set -u
LOG=/home/zhuojun/prof/launchblock.log
for i in $(seq 1 100); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  grep -qE 'FATAL|LAUNCHBLOCK_SERVE_EXIT' "$LOG" 2>/dev/null && break
  sleep 3
done
grep 'listening on' "$LOG" | tail -1
timeout 300 curl -s --max-time 280 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/pelican_none_req.json -o /home/zhuojun/prof/lb2.json \
  -w 'lb_pelican http=%{http_code} wall=%{time_total}s\n' || echo 'curl timeout/failed'
sleep 4
echo '--- error lines ---'
grep -nE 'failed:|error:|LAUNCHBLOCK_SERVE_EXIT|terminate' "$LOG" | tail -6
echo '--- last mtp rounds ---'
grep 'mtp. round' "$LOG" | tail -3
