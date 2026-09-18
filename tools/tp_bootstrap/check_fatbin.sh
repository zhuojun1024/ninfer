#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
echo "=== SERVE fatbin section ==="
readelf -S $B/apps/ninfer-serve 2>/dev/null | grep -i "fatbin\|nv_"
echo "=== TEST fatbin section ==="
readelf -S $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -i "fatbin\|nv_"
echo "=== SERVE .nv_fatbin size ==="
readelf -S $B/apps/ninfer-serve 2>/dev/null | grep -i "fatbin" | awk '{print $NF}'
echo "=== TEST .nv_fatbin size ==="
readelf -S $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -i "fatbin" | awk '{print $NF}'
