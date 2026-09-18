#!/bin/bash
set -e
cp /mnt/d/Documents/workbench/ninfer/src/ops/linear/linear.cpp /home/zhuojun/ninfer/src/ops/linear/linear.cpp
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build --target ninfer_qwen3_5_tp2_forward_test -j 8 2>&1 | tail -3
echo "BUILD_EXIT=${PIPESTATUS[0]}"
./build/tests/ninfer_qwen3_5_tp2_forward_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer 2>&1 | tail -15
echo "TEST_EXIT=${PIPESTATUS[0]}"
