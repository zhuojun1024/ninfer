#!/bin/bash
# Run the crashing MTP request under compute-sanitizer to localize the illegal address.
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
LOG=/home/zhuojun/prof/sanitize.log
: > "$LOG"
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
echo "sanitizer: $(command -v compute-sanitizer)" | tee -a "$LOG"
compute-sanitizer --tool memcheck --launch-timeout 600 --print-limit 2 \
  ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8089 --spec mtp --draft-tokens 2 \
  --temperature 0.7 --top-k 20 --top-p 0.80 >> "$LOG" 2>&1
echo "SANITIZED_SERVE_EXIT=$?" >> "$LOG"
