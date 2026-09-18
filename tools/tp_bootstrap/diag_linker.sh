#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
CUDALIB=/usr/local/cuda-13.1/targets/x86_64-linux/lib
echo '=== A) bare g++ minlink, NO fatbin objects ==='
g++ /tmp/minlink.cpp -o /tmp/minlink_bare -lcudart -L $CUDALIB
timeout 20 /tmp/minlink_bare; echo EXIT_BARE=$?
echo '=== B) obj_device.o linked with nvcc driver (not g++) ==='
nvcc /tmp/minlink.cpp /tmp/obj_device.o -o /tmp/minlink_nvcc -lcudart -L $CUDALIB 2>/dev/null
timeout 20 /tmp/minlink_nvcc; echo EXIT_NVCC=$?
echo '=== C) main ninfer CLI: does it reach CUDA init? (bogus model path) ==='
timeout 20 ./build/apps/ninfer /nonexistent.ninfer 2>&1 | head -3; echo EXIT_CLI=$?