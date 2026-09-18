#!/bin/bash
# Leave one normal (non-greedy) 64k serve running for manual testing.
set -u
cd /home/zhuojun/ninfer
LOG=/home/zhuojun/prof/serve_final.log
HEALTH=http://127.0.0.1:8088/health

pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null
for _ in $(seq 1 30); do pgrep -f 'build/apps/ninfer-serve' >/dev/null || break; sleep 1; done
if pgrep -f 'build/apps/ninfer-serve' >/dev/null; then echo 'REFUSING: serve still alive'; pgrep -a -f 'build/apps/ninfer-serve'; exit 3; fi
free -h | head -2
setsid nohup ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8088 < /dev/null > "$LOG" 2>&1 &
echo "launched"
ready=0
for i in $(seq 1 150); do
  if grep -q 'listening on' "$LOG" 2>/dev/null; then
    if [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "$HEALTH" 2>/dev/null)" = "200" ]; then
      echo "healthy after ${i}s"; ready=1; break
    fi
  fi
  sleep 1
done
if [ "$ready" != "1" ]; then echo 'NOT HEALTHY'; tail -20 "$LOG"; exit 4; fi
echo '--- exactly one serve ---'
pgrep -a -f 'apps/ninfer-serve'
free -h | head -2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo '--- smoke request ---'
curl -s --max-time 120 -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' -o /tmp/smoke.json -w 'http=%{http_code} total=%{time_total}s\n' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Two plus two? Answer with the number only."}],"max_tokens":64,"stream":false}'
python3 -c "
import json
d=json.load(open('/tmp/smoke.json'))
m=d['choices'][0]['message']
print('content:', repr(m.get('content'))[:200])
print('reason:', repr(m.get('reasoning_content'))[:200])
"
grep -E 'listening|engine ready|done' "$LOG" | tail -4
