#!/bin/bash
set -e
cp /mnt/d/Documents/workbench/ninfer/src/models/qwen3_5/load/tp_split_spec.cpp /home/zhuojun/ninfer/src/models/qwen3_5/load/tp_split_spec.cpp
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -3
echo "BUILD_EXIT=${PIPESTATUS[0]}"
./build/tests/ninfer_qwen3_5_tp2_forward_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer > /tmp/tp2fwd2.log 2>&1
echo "TEST_EXIT=$?"
echo "=== lines of interest ==="
grep -E 'linear-diag|FAIL|PASS|probes|shard state|OK|mismatch|max' /tmp/tp2fwd2.log | head -40
