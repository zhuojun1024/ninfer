#!/usr/bin/env bash
B=/home/zhuojun/ninfer/build
MANGLED=_ZN6ninfer3ops21set_i32_scalar_kernelEPii
echo "=== SERVE: host symbol state ==="
nm -C $B/apps/ninfer-serve 2>/dev/null | grep "set_i32_scalar_kernel" | head
echo "--- raw (mangled) ---"
nm $B/apps/ninfer-serve 2>/dev/null | grep "set_i32_scalar_kernel" | head
echo ""
echo "=== TEST: host symbol state ==="
nm -C $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep "set_i32_scalar_kernel" | head
echo "--- raw (mangled) ---"
nm $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep "set_i32_scalar_kernel" | head
echo ""
echo "=== SERVE: readelf dynsym for set_i32 ==="
readelf --dyn-syms $B/apps/ninfer-serve 2>/dev/null | grep -i "set_i32" | head
echo "=== TEST: readelf dynsym for set_i32 ==="
readelf --dyn-syms $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -i "set_i32" | head
