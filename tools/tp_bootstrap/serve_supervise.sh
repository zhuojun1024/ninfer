#!/bin/bash
# Keep exactly one ninfer-serve alive AND keep this WSL session open.
#
# Two jobs at once:
#  1. the WSL distribution powers itself off (systemctl poweroff) when its last session ends, which
#     kills any detached serve; holding a foreground session here prevents that;
#  2. if the engine dies (crash, bad input), restart it instead of leaving the user with a dead port.
set -u
CTX="${1:-65536}"
PORT="${2:-8088}"
EXTRA="${3:-}"
PATTERN='build/apps/ninfer-serve'
LOG=/home/zhuojun/prof/serve_supervised.log
cd /home/zhuojun/ninfer
echo "=== supervisor start $(date +%H:%M:%S) ctx=$CTX port=$PORT log=$LOG ==="
while true; do
  pkill -9 -f "$PATTERN" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "$PATTERN" >/dev/null || break; sleep 1; done
  if pgrep -f "$PATTERN" >/dev/null; then
    echo "=== REFUSING to start: another serve is still alive $(date +%H:%M:%S) ==="
    pgrep -a -f "$PATTERN"
    sleep 15
    continue
  fi
  free -h | head -2
  echo "=== launch $(date +%H:%M:%S) ==="
  # shellcheck disable=SC2086
  ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
    --devices 0,1 --max-context "$CTX" --port "$PORT" $EXTRA 2>&1 | tee -a "$LOG"
  code=${PIPESTATUS[0]}
  echo "=== serve exited code=$code at $(date +%H:%M:%S); restarting in 5s ==="
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  sleep 5
done
