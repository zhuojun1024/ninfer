#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
echo "=== SERVE: set_i32_scalar_kernel SASS ==="
cuobjdump --dump-sass $B/apps/ninfer-serve 2>/dev/null | grep "set_i32_scalar_kernel" | head -3
echo "count: $(cuobjdump --dump-sass $B/apps/ninfer-serve 2>/dev/null | grep -c 'set_i32_scalar_kernel')"
echo "=== TEST: set_i32_scalar_kernel SASS ==="
cuobjdump --dump-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep "set_i32_scalar_kernel" | head -3
echo "count: $(cuobjdump --dump-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -c 'set_i32_scalar_kernel')"
echo "=== SERVE: all scalar kernels ==="
cuobjdump --dump-sass $B/apps/ninfer-serve 2>/dev/null | grep -oE "Function : [^"]*scalar[^"]*" | sort -u | head -20
echo "=== TEST: all scalar kernels ==="
cuobjdump --dump-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -oE "Function : [^"]*scalar[^"]*" | sort -u | head -20
