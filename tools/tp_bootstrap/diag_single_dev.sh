#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
LIB=build/src/core/libninfer_core.a
g++ /tmp/minlink.cpp -o /tmp/minlink -L $(dirname $LIB) -lninfer_core -lcudart -L /usr/local/cuda-13.1/targets/x86_64-linux/lib 2>/dev/null
echo '=== t1 sanity: does WSL honor CUDA_VISIBLE_DEVICES? ==='
CUDA_VISIBLE_DEVICES=0,2 /tmp/t1 2>/dev/null | head -3
echo '=== minlink on 5060 Ti only (dev 0) ==='
CUDA_VISIBLE_DEVICES=0 timeout 20 /tmp/minlink; echo EXIT_DEV0=$?
echo '=== minlink on T10 only (dev 1) ==='
CUDA_VISIBLE_DEVICES=1 timeout 20 /tmp/minlink; echo EXIT_DEV1=$?
echo '=== fatbin PTX in libninfer_core.a ==='
cuobjdump -ptx $LIB 2>/dev/null | grep -E 'setmaxnreg|target|address_space|.visible' | head -30