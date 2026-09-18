#!/bin/bash
set -e
cp /mnt/d/Documents/workbench/ninfer/tests/models/qwen3_5/test_tp2_load.cpp /home/zhuojun/ninfer/tests/models/qwen3_5/test_tp2_load.cpp
cp /mnt/d/Documents/workbench/ninfer/tests/models/qwen3_5/test_tp2_forward.cpp /home/zhuojun/ninfer/tests/models/qwen3_5/test_tp2_forward.cpp
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -25
echo "BUILD_EXIT=${PIPESTATUS[0]}"
