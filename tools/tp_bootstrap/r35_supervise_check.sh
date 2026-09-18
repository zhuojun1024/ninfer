#!/bin/bash
sleep 45
echo '--- serve processes ---'
pgrep -a -f 'apps/ninfer-serve' || echo 'NO serve process'
echo '--- health ---'
curl -s --max-time 5 -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8088/health
echo '--- gpu / memory ---'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
free -h | head -2
echo '--- supervisor log tail ---'
tail -8 /home/zhuojun/prof/serve_supervised.log 2>/dev/null
echo '--- supervised smoke ---'
curl -s --max-time 120 -X POST http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -o /tmp/smoke3.json -w 'http=%{http_code} total=%{time_total}s\n' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":32,"stream":false}'
python3 -c "
import json; d=json.load(open('/tmp/smoke3.json')); m=d['choices'][0]['message']
print('content:', repr(m.get('content'))[:160])
" 2>/dev/null || echo 'no json yet'
