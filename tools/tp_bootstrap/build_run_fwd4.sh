#!/bin/bash
set -e
cp -r /mnt/d/Documents/workbench/ninfer/src/ops/softmax_attention/dense/causal_cache/. /home/zhuojun/ninfer/src/ops/softmax_attention/dense/causal_cache/
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -6
echo "BUILD_EXIT=${PIPESTATUS[0]}"
./build/tests/ninfer_qwen3_5_tp2_forward_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer > /tmp/tp2fwd4.log 2>&1
echo "TEST_EXIT=$?"
tail -14 /tmp/tp2fwd4.log
