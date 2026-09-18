#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
LIB=build/src/core/libninfer_core.a
echo '=== SASS function count (cubin present?) ==='
cuobjdump -sass $LIB 2>&1 | grep -c 'Function :'
echo '=== fatbin sections ==='
cuobjdump $LIB 2>&1 | head -20
echo '=== PTX: unusual instructions ==='
cuobjdump -ptx $LIB 2>/dev/null | grep -E 'setmaxnreg|griddepcontrol|fence|redux|cluster' | sort | uniq -c | head -20
echo '=== CMake cuda archive function ==='
grep -rn 'ninfer_cuda_archive' cmake/ CMakeLists.txt 2>/dev/null | head -5