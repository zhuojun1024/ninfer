#!/bin/bash
set -e
cp /mnt/d/Documents/workbench/ninfer/src/ops/linear/linear.cpp /home/zhuojun/ninfer/src/ops/linear/linear.cpp
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -4
echo "BUILD_EXIT=${PIPESTATUS[0]}"
echo "=== load test ==="
./build/tests/ninfer_qwen3_5_tp2_load_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer 2>&1 | tail -6
echo "=== forward test ==="
./build/tests/ninfer_qwen3_5_tp2_forward_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer > /tmp/tp2fwd5.log 2>&1
echo "FORWARD_EXIT=$?"
tail -3 /tmp/tp2fwd5.log
ls -la --time-style=+%H:%M build/apps/ninfer-serve
