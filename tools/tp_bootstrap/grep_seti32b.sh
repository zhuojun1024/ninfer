#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
S=$(cuobjdump --dump-sass $B/apps/ninfer-serve 2>/dev/null)
T=$(cuobjdump --dump-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null)
echo "SERVE total SASS funcs: $(echo "$S" | grep -cE 'Function :')"
echo "SERVE set_i32_scalar_kernel hits: $(echo "$S" | grep -c 'set_i32_scalar_kernel')"
echo "SERVE set_i32 lines:"
echo "$S" | grep 'set_i32_scalar_kernel' | head -5
echo ""
echo "TEST total SASS funcs: $(echo "$T" | grep -cE 'Function :')"
echo "TEST set_i32_scalar_kernel hits: $(echo "$T" | grep -c 'set_i32_scalar_kernel')"
echo "TEST set_i32 lines:"
echo "$T" | grep 'set_i32_scalar_kernel' | head -5
