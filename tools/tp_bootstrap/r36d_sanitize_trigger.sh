#!/bin/bash
# Wait for the sanitized serve, fire the crashing request, dump the first sanitizer error.
set -u
LOG=/home/zhuojun/prof/sanitize.log
for i in $(seq 1 240); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  grep -qE 'FATAL|SANITIZED_SERVE_EXIT' "$LOG" 2>/dev/null && break
  sleep 3
done
grep -E 'listening on' "$LOG" | tail -1
curl -s --max-time 1800 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a detailed paragraph about water cycle."}],"max_tokens":900}' \
  -o /home/zhuojun/prof/sanitize_repro.json -w 'sanitize_repro http=%{http_code} wall=%{time_total}s\n'
sleep 5
echo '--- last rounds ---'
grep -c '\[mtp\] round' "$LOG"
grep '\[mtp\] round' "$LOG" | tail -3
echo '--- first sanitizer error block ---'
awk '/Invalid|out of bounds|Address 0x/{found=1} found{print; n++} n>28{exit}' "$LOG" | head -40
echo '--- tail ---'
tail -12 "$LOG"
