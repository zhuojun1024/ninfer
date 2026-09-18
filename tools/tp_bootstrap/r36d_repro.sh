#!/bin/bash
# Try to reproduce the illegal-address crash: the exact request that killed the server before,
# three times, checking liveness each time.
set -u
cd /home/zhuojun/prof
BODY='{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a detailed paragraph about water cycle."}],"max_tokens":900}'
for i in 1 2 3; do
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$BODY" -o "repro_$i.json" -w "try$i http=%{http_code} wall=%{time_total}s\n"
  sleep 3
  if [ -s "repro_$i.json" ]; then
    python3 -c "import json; d=json.load(open('repro_$i.json')); u=d.get('usage',{}); t=d.get('timings',{}); print('   ok tokens=%s %.1f tok/s finish=%s' % (u.get('completion_tokens'), t.get('predicted_per_second',-1), d['choices'][0].get('finish_reason')))"
  else
    echo "   EMPTY -> server died"
    tail -12 serve_supervised.log
    break
  fi
done
echo "--- alive? ---"; pgrep -af ninfer-serve | head -1
echo "--- log tail ---"; tail -8 serve_supervised.log
