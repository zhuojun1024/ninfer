#!/bin/bash
# Plain (non-MTP) serve with the same sampling defaults, logging to its own file and holding WSL.
set -u
LOG=/home/zhuojun/prof/plain_pelican.log
cd /home/zhuojun/ninfer
exec ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8089 \
  --temperature 0.7 --top-k 20 --top-p 0.80 >> "$LOG" 2>&1
