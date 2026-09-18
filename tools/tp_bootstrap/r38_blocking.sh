#!/bin/bash
# Localize the batched-verify fault with CUDA_LAUNCH_BLOCKING=1.
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
log=/home/zhuojun/prof/blocking.log
: > "$log"
CUDA_LAUNCH_BLOCKING=1 ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8088 --spec mtp --draft-tokens 2 >> "$log" 2>&1 &
for i in $(seq 1 420); do
  grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break
  pgrep -f 'build/apps/ninfer-serve' >/dev/null || break
  sleep 2
done
if grep -q 'listening on http://127.0.0.1:8088' "$log"; then
  curl -s --max-time 240 http://127.0.0.1:8088/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @/home/zhuojun/prof/trace_req.json \
    -o /home/zhuojun/prof/blocking.json -w "blocking http=%{http_code}\n" >> "$log" 2>&1
fi
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
out=/home/zhuojun/prof/blocking-out.log
: > "$out"
grep -anE 'failed:|FATAL|CUDA_CHECK|\.cu:[0-9]+:' "$log" | tail -8 >> "$out"
grep -ac 'listening on' "$log" >> "$out"
echo BLOCKING_DONE >> "$out"
