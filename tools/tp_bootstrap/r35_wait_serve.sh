#!/bin/bash
LOG=/home/zhuojun/prof/serve_restart_probe.log
echo '--- wait for serve ---'
ok=0
for i in $(seq 1 90); do
  if [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null)" = "200" ]; then
    echo "healthy after ${i}s"; ok=1; break
  fi
  sleep 1
done
[ "$ok" = "1" ] || { echo 'NOT HEALTHY YET'; pgrep -a -f 'apps/ninfer-serve' || echo 'no serve process'; exit 4; }
echo '--- process / gpu / memory ---'
pgrep -a -f 'apps/ninfer-serve'
free -h | head -2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo '--- smoke ---'
curl -s --max-time 120 -X POST http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -o /tmp/smoke2.json -w 'http=%{http_code} total=%{time_total}s\n' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Name the largest planet. One word."}],"max_tokens":48,"stream":false}'
python3 -c "
import json; d=json.load(open('/tmp/smoke2.json')); m=d['choices'][0]['message']
print('content:', repr(m.get('content'))[:160])
print('reason:', repr(m.get('reasoning_content'))[:160])
"
