#!/bin/bash
# A/B the TP-2 allreduce transport at prefill chunk 1024: in-kernel host staging vs copy engines.
set -u
cd /home/zhuojun/ninfer
PORT=8099
LOG=/home/zhuojun/prof/ab_sweep_serve.log
mkdir -p /home/zhuojun/prof
LONG=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 120))
printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s Count the sentences."}],"max_tokens":4,"stream":false}' "$LONG" > /home/zhuojun/prof/sweep_req.json
for M in default kernel copies; do
  pkill -9 -f 'ninfer-serve' 2>/dev/null
  for _ in $(seq 1 20); do pgrep -f 'ninfer-serve' >/dev/null || break; sleep 1; done
  : > "$LOG"
  if [ "$M" = "default" ]; then ENVV=""; else ENVV="NINFER_AR_MODE=$M"; fi
  env $ENVV setsid nohup ./build/apps/ninfer-serve \
    /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 --max-context 8192 \
    --port $PORT --prefill-chunk 1024 --no-prefix-reuse < /dev/null >> "$LOG" 2>&1 &
  ready=0
  for i in $(seq 1 90); do
    if grep -q 'listening on' "$LOG" 2>/dev/null && \
       [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ]; then ready=1; break; fi
    sleep 1
  done
  if [ "$ready" != "1" ]; then echo "MODE=$M SERVE NOT READY"; tail -5 "$LOG"; continue; fi
  for run in 1 2; do
    curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      --data-binary @/home/zhuojun/prof/sweep_req.json -o /dev/null -w "MODE=$M run=$run total=%{time_total}s "
    grep -E 'req#1 done' "$LOG" | tail -1
  done
  pkill -9 -f 'ninfer-serve' 2>/dev/null
  sleep 2
done
echo '--- done ---'
