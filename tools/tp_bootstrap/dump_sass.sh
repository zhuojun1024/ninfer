#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
echo "=== SERVE: total SASS function entries ==="
cuobjdump --dump-sass $B/apps/ninfer-serve 2>/dev/null | grep -cE "^\s+Function"
echo "=== SERVE: sample function lines ==="
cuobjdump --dump-sass $B/apps/ninfer-serve 2>/dev/null | grep -E "^\s+Function" | head -15
echo "=== SERVE: set_i32 mangled ==="
cuobjdump --dump-sass $B/apps/ninfer-serve 2>/dev/null | grep -i "set_i32" | head
echo "=== TEST: total SASS function entries ==="
cuobjdump --dump-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -cE "^\s+Function"
echo "=== TEST: sample function lines ==="
cuobjdump --dump-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -E "^\s+Function" | head -15
echo "=== TEST: set_i32 mangled ==="
cuobjdump --dump-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -i "set_i32" | head
