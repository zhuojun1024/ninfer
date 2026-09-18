#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
LIB=build/src/core/libninfer_core.a
g++ /tmp/minlink.cpp -o /tmp/minlink -L $(dirname $LIB) -lninfer_core -lcudart -L /usr/local/cuda-13.1/targets/x86_64-linux/lib 2>/dev/null
echo '=== minlink with all 3 devices visible ==='
timeout 20 /tmp/minlink; echo EXIT_ALL=$?
echo '=== minlink with CUDA_VISIBLE_DEVICES=0,2 (T10 hidden) ==='
CUDA_VISIBLE_DEVICES=0,2 timeout 20 /tmp/minlink; echo EXIT_HIDDEN=$?
echo '=== tp test with CUDA_VISIBLE_DEVICES=0,2 ==='
CUDA_VISIBLE_DEVICES=0,2 timeout 60 ./build/tests/ninfer_tp_device_pair_test; echo TP_TEST_EXIT=$?