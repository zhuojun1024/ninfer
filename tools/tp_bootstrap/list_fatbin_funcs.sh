#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
echo "=== SERVE: fatbin function names (mangled) ==="
cuobjdump --dump-elf $B/apps/ninfer-serve 2>/dev/null | grep -oE "Function : [^ ]+" | sort -u | head -40
echo "count:"
cuobjdump --dump-elf $B/apps/ninfer-serve 2>/dev/null | grep -c "Function :"
echo "=== TEST: fatbin function names ==="
cuobjdump --dump-elf $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -oE "Function : [^ ]+" | sort -u | head -40
echo "count:"
cuobjdump --dump-elf $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -c "Function :"
echo "=== SERVE: scalar-ish mangled names ==="
cuobjdump --dump-elf $B/apps/ninfer-serve 2>/dev/null | grep -oE "Function : [^ ]+" | grep -i "scalar\|set_i32\|i32" | head
echo "=== TEST: scalar-ish mangled names ==="
cuobjdump --dump-elf $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -oE "Function : [^ ]+" | grep -i "scalar\|set_i32\|i32" | head
