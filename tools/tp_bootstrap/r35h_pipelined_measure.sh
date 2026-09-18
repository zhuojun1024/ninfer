#!/bin/bash
# Measure the pipelined allreduce at prefill chunk 1024: fresh 1257-token prompt, no prefix reuse.
set -u
cd /home/zhuojun/ninfer
PORT=8099
LOG=/home/zhuojun/prof/pipelined_measure.log
LONG=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 120))
printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s Count the sentences."}],"max_tokens":4,"stream":false}' "$LONG" > /home/zhuojun/prof/sweep_req.json
: > "$LOG"
setsid nohup ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 \
  --max-context 8192 --port $PORT --prefill-chunk 1024 --no-prefix-reuse < /dev/null >> "$LOG" 2>&1 &
for i in $(seq 1 120); do
  if grep -q 'listening on' "$LOG" 2>/dev/null && \
     [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ]; then break; fi
  sleep 1
done
for run in 1 2 3; do
  curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/sweep_req.json -o /dev/null -w "run=$run wall=%{time_total}s\n"
  grep -E 'req#1 done' "$LOG" | tail -1
done
pkill -9 -f 'ninfer-serve' 2>/dev/null
sleep 2
echo '--- done ---'
