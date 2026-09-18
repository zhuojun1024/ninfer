#!/bin/bash
# Debug build: bisect the MTP fault phase by firing the crashing request twice.
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
LOG=/home/zhuojun/prof/dbg.log
: > "$LOG"
cd /home/zhuojun/ninfer
./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8088 --spec mtp --draft-tokens 2 \
  --temperature 0.7 --top-k 20 --top-p 0.80 >> "$LOG" 2>&1 &
SERVE=$!
for i in $(seq 1 200); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  kill -0 $SERVE 2>/dev/null || break
  sleep 2
done
echo "=== ready (alive=$(kill -0 $SERVE 2>/dev/null && echo yes || echo no)) ===" >> "$LOG"
for attempt in 1 2 3; do
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/pelican_none_req.json -o "/home/zhuojun/prof/dbg_$attempt.json" \
    -w "DBG attempt$attempt http=%{http_code} wall=%{time_total}s\n" >> "$LOG" 2>&1 || echo "curl fail $attempt" >> "$LOG"
  sleep 3
  kill -0 $SERVE 2>/dev/null || { echo "=== serve died after attempt$attempt ===" >> "$LOG"; break; }
done
echo "=== DBG_DONE ===" >> "$LOG"
wait $SERVE
echo "=== SERVE_EXIT=$? ===" >> "$LOG"
