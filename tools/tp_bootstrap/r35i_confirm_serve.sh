#!/bin/bash
for i in $(seq 1 120); do
  if grep -q 'listening on' /home/zhuojun/prof/serve_supervised.log 2>/dev/null; then break; fi
  sleep 2
done
echo '--- procs ---'; pgrep -af 'ninfer-serve|serve_supervise'; echo '--- health ---'
curl -s --max-time 5 -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8088/health
echo '--- warm smoke ---'
curl -s --max-time 120 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"reply with the single word: ok"}],"max_tokens":4}' \
  -o /tmp/warm.json -w 'http=%{http_code} wall=%{time_total}s\n'
grep -E 'req#1 done|throughput' /home/zhuojun/prof/serve_supervised.log | tail -2
free -g | head -2
