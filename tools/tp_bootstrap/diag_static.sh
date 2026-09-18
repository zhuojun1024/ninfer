#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
CUDALIB=/usr/local/cuda-13.1/targets/x86_64-linux/lib

echo '=== A) nvcc -cudart=shared (force dynamic) on minlink.cpp ==='
nvcc -cudart=shared /tmp/minlink.cpp -o /tmp/minlink_shared -lcudart -L $CUDALIB 2>/dev/null
ldd /tmp/minlink_shared | grep cudart
timeout 20 /tmp/minlink_shared >/dev/null 2>&1 && echo 'A: PASS' || echo 'A: CRASH'

echo '=== B) nvcc -cudart=static (force static) on minlink.cpp ==='
nvcc -cudart=static /tmp/minlink.cpp -o /tmp/minlink_static -L $CUDALIB 2>/dev/null
ldd /tmp/minlink_static | grep cudart || echo '(no dynamic cudart = static)'
timeout 20 /tmp/minlink_static >/dev/null 2>&1 && echo 'B: PASS' || echo 'B: CRASH'

echo '=== C) how does the ninfer CLI link cudart? ==='
ldd ./build/apps/ninfer | grep -E 'cudart|cuda'

echo '=== D) CMake CUDA runtime library setting ==='
grep -rn 'CUDA_RUNTIME_LIBRARY\|cudart\|CMAKE_CUDA_RUNTIME' CMakeLists.txt cmake/ 2>/dev/null | head -10