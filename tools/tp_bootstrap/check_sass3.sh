#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
echo "=== SERVE SASS function count ==="
cuobjdump --list-sass $B/apps/ninfer-serve 2>/dev/null | grep -c "Function"
echo "=== SERVE set_i32 in SASS ==="
cuobjdump --list-sass $B/apps/ninfer-serve 2>/dev/null | grep -i "set_i32" | head
echo "=== TEST SASS function count ==="
cuobjdump --list-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -c "Function"
echo "=== TEST set_i32 in SASS ==="
cuobjdump --list-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -i "set_i32" | head
