#!/bin/bash
set -e
cd /mnt/d/Documents/workbench/ninfer
for f in src/ops/linear/fp8/fp8_shapes.h src/ops/linear/fp8/fp8_dispatch.cpp src/ops/linear/fp8/sources.cmake src/ops/linear/fp8/shapes/n5120_k3072.cu; do
  cp "$f" "/home/zhuojun/ninfer/$f"
done
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -6
echo "BUILD_EXIT=${PIPESTATUS[0]}"
./build/tests/ninfer_qwen3_5_tp2_forward_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer > /tmp/tp2fwd3.log 2>&1
echo "TEST_EXIT=$?"
tail -12 /tmp/tp2fwd3.log
