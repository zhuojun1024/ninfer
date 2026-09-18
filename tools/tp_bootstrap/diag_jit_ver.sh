#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
LIB=build/src/core/libninfer_core.a
g++ /tmp/minlink.cpp -o /tmp/minlink -L $(dirname $LIB) -lninfer_core -lcudart -L /usr/local/cuda-13.1/targets/x86_64-linux/lib 2>/dev/null

echo '=== 1) CUDA_DISABLE_PTX_JIT_COMPILATION=1 (force cubin-only) ==='
CUDA_DISABLE_PTX_JIT_COMPILATION=1 timeout 20 /tmp/minlink 2>&1; echo EXIT_NOJIT=$?

echo '=== 2) LD_LIBRARY_PATH with CUDA 13.1 ptxjitcompiler ==='
PTXJIT=$(find /usr/local/cuda-13.1 -name 'libnvidia-ptxjitcompiler*' 2>/dev/null | head -1)
echo "ptxjit found: $PTXJIT"
if [ -n "$PTXJIT" ]; then
  LD_LIBRARY_PATH=$(dirname $PTXJIT):$LD_LIBRARY_PATH timeout 20 /tmp/minlink 2>&1; echo EXIT_131JIT=$?
fi

echo '=== 3) Check which ptxjitcompiler the 13.1 runtime expects ==='
ldd /usr/local/cuda-13.1/targets/x86_64-linux/lib/libcudart.so.13 2>/dev/null | grep -i ptxjit
echo '--- system ptxjitcompiler version ---'
strings /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 2>/dev/null | grep -i 'cuda\|version\|12\.' | head -5