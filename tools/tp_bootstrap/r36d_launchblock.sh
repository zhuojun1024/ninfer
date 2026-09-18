#!/bin/bash
# Localize the illegal address by making every launch synchronous, so the failing launch reports
# its own CUDA_CHECK site instead of the next sync.
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f compute-sanitizer 2>/dev/null || true
LOG=/home/zhuojun/prof/launchblock.log
: > "$LOG"
cd /home/zhuojun/ninfer
export CUDA_LAUNCH_BLOCKING=1
./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8089 --spec mtp --draft-tokens 2 \
  --temperature 0.7 --top-k 20 --top-p 0.80 >> "$LOG" 2>&1
echo "LAUNCHBLOCK_SERVE_EXIT=$?" >> "$LOG"
