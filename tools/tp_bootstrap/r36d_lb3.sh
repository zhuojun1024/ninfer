#!/bin/bash
# Self-contained: start the MTP serve with CUDA_LAUNCH_BLOCKING, then fire the crashing request.
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
LOG=/home/zhuojun/prof/launchblock.log
: > "$LOG"
cd /home/zhuojun/ninfer
export CUDA_LAUNCH_BLOCKING=1
./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8089 --spec mtp --draft-tokens 2 \
  --temperature 0.7 --top-k 20 --top-p 0.80 >> "$LOG" 2>&1 &
SERVE=$!
for i in $(seq 1 300); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  kill -0 $SERVE 2>/dev/null || break
  sleep 2
done
echo "=== readiness loop exited; serve alive=$(kill -0 $SERVE 2>/dev/null && echo yes || echo no) ===" >> "$LOG"
curl -s --max-time 300 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/pelican_none_req.json -o /home/zhuojun/prof/lb3.json \
  -w 'LB3 http=%{http_code} wall=%{time_total}s\n' >> "$LOG" 2>&1 || echo 'curl failed' >> "$LOG"
sleep 5
echo "=== LB3_DONE ===" >> "$LOG"
wait $SERVE
echo "=== SERVE_EXIT=$? ===" >> "$LOG"
