#!/bin/bash
# Launch exactly one ninfer-serve. A leftover instance must be gone first: two serves each stage
# the 22.6 GB artifact in host memory and double-book both GPUs.
set -u
CTX="${1:-65536}"
PATTERN='build/apps/ninfer-serve'

pkill -9 -f "$PATTERN" 2>/dev/null
for _ in $(seq 1 30); do
  pgrep -f "$PATTERN" >/dev/null || break
  sleep 1
done
if pgrep -f "$PATTERN" >/dev/null; then
  echo 'REFUSING to start: another ninfer-serve is still alive'; pgrep -a -f "$PATTERN"; exit 3
fi
echo 'host memory before launch:'; free -h | head -2
cd /home/zhuojun/ninfer
echo "serve start $(date +%H:%M:%S) max-context=$CTX"
exec ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 --max-context $CTX --port 8088