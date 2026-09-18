#!/bin/bash
# Experiment: sweep the in-kernel allreduce block count (NINFER_AR_BLOCKS) at prefill chunk 1024.
set -u
cd /home/zhuojun/ninfer
PORT=8099
LOG=/home/zhuojun/prof/ar_sweep_serve.log
mkdir -p /home/zhuojun/prof
LONG=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 120))
printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s Count the sentences."}],"max_tokens":4,"stream":false}' "$LONG" > /home/zhuojun/prof/sweep_req.json
for B in 0 1 2 4 8 16; do
  pkill -9 -f 'ninfer-serve' 2>/dev/null
  for _ in $(seq 1 20); do pgrep -f 'ninfer-serve' >/dev/null || break; sleep 1; done
  : > "$LOG"
  NINFER_AR_BLOCKS=$B setsid nohup ./build/apps/ninfer-serve \
    /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 --max-context 8192 \
    --port $PORT --prefill-chunk 1024 --no-prefix-reuse < /dev/null >> "$LOG" 2>&1 &
  ready=0
  for i in $(seq 1 90); do
    if grep -q 'listening on' "$LOG" 2>/dev/null && \
       [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ]; then ready=1; break; fi
    sleep 1
  done
  if [ "$ready" != "1" ]; then echo "B=$B SERVE NOT READY"; tail -5 "$LOG"; continue; fi
  curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/sweep_req.json -o /dev/null -w "AR_BLOCKS=$B total=%{time_total}s "
  grep -E 'req#1 done' "$LOG" | tail -1
  echo "   log: $(grep -c '\[ar\]' "$LOG") diag lines: $(grep -m2 '\[ar\]' "$LOG" | tr '\n' ' ')"
  pkill -9 -f 'ninfer-serve' 2>/dev/null
  sleep 2
done
echo '--- done ---'
