#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
echo "=== 1) ninfer CLI --help (main build binary) ==="
timeout 30 ./build/apps/ninfer --help 2>&1 | head -5
echo "CLI_EXIT=$?"
echo "=== 2) tp test with CUDA_MODULE_LOADING=EAGER ==="
CUDA_MODULE_LOADING=EAGER timeout 30 ./build/tests/ninfer_tp_device_pair_test 2>&1 | head -10
echo "EAGER_EXIT=$?"
echo "=== 3) tp test with CUDA_MODULE_LOADING=LAZY ==="
CUDA_MODULE_LOADING=LAZY timeout 30 ./build/tests/ninfer_tp_device_pair_test 2>&1 | head -10
echo "LAZY_EXIT=$?"
