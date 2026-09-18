#!/bin/bash
# Prefill chunk sweep: for each --prefill-chunk width, start one serve (port 8099) and time a fresh
# ~1750-token prompt with prefix reuse disabled, so every request is a full prefill.
set -u
cd /home/zhuojun/ninfer
PORT=8099
HEALTH=http://127.0.0.1:$PORT/health
URL=http://127.0.0.1:$PORT/v1/chat/completions
LOG=/home/zhuojun/prof/sweep_serve.log
mkdir -p /home/zhuojun/prof

LONG=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 120))
printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s Count the sentences."}],"max_tokens":4,"stream":false}' "$LONG" > /home/zhuojun/prof/sweep_req.json

for CHUNK in 128 256 512 1024; do
  pkill -9 -f 'ninfer-serve' 2>/dev/null
  for _ in $(seq 1 20); do pgrep -f 'ninfer-serve' >/dev/null || break; sleep 1; done
  : > "$LOG"
  setsid nohup ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
    --devices 0,1 --max-context 8192 --port $PORT --prefill-chunk $CHUNK --no-prefix-reuse \
    < /dev/null >> "$LOG" 2>&1 &
  ready=0
  for i in $(seq 1 90); do
    if grep -q 'listening on' "$LOG" 2>/dev/null; then
      if [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "$HEALTH" 2>/dev/null)" = "200" ]; then
        ready=1; break
      fi
    fi
    sleep 1
  done
  if [ "$ready" != "1" ]; then echo "chunk=$CHUNK SERVE NOT READY"; tail -10 "$LOG"; continue; fi
  curl -s --max-time 300 -X POST "$URL" -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/sweep_req.json -o /dev/null -w "chunk=$CHUNK http=%{http_code} total=%{time_total}s "
  grep -E 'req#1 done' "$LOG" | tail -1
  pkill -9 -f 'ninfer-serve' 2>/dev/null
  sleep 2
done
echo '--- done ---'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
