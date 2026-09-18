#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
echo '=== ldd of rebuilt tp test (should be NO dynamic cudart if static) ==='
ldd ./build/tests/ninfer_tp_device_pair_test | grep -E 'cudart|cuda|ptxjit' || echo '(none = fully static)'

echo '=== fatbin count in tp test ==='
cuobjdump ./build/tests/ninfer_tp_device_pair_test 2>/dev/null | grep -c 'Fatbin'

echo '=== gdb backtrace of tp test now ==='
gdb -batch -ex run -ex bt --args ./build/tests/ninfer_tp_device_pair_test 2>&1 | tail -25