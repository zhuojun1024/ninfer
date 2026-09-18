#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
echo "=== SERVE: scalar kernel device functions ==="
cuobjdump --list-elf $B/apps/ninfer-serve 2>/dev/null | grep -i "scalar" | head
echo "serve total device functions:"
cuobjdump --list-elf $B/apps/ninfer-serve 2>/dev/null | grep -c "Function"
echo "=== TEST: scalar kernel device functions ==="
cuobjdump --list-elf $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -i "scalar" | head
echo "test total device functions:"
cuobjdump --list-elf $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -c "Function"
echo "=== SERVE: any set_i32 at all (SASS) ==="
cuobjdump $B/apps/ninfer-serve 2>/dev/null | grep -c "set_i32_scalar_kernel"
echo "=== TEST: any set_i32 at all (SASS) ==="
cuobjdump $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -c "set_i32_scalar_kernel"
