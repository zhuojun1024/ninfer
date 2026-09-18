#!/usr/bin/env bash
B=/home/zhuojun/ninfer/build
echo "=== SERVE __nv_module_id section hex ==="
readelf -x __nv_module_id $B/apps/ninfer-serve 2>/dev/null
echo "=== TEST __nv_module_id section hex ==="
readelf -x __nv_module_id $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null
echo ""
cd /tmp && rm -rf midcheck && mkdir midcheck && cd midcheck
echo "=== ops device_link.o __nv_module_id ==="
ar x $B/src/ops/libninfer_ops.a cmake_device_link.o 2>/dev/null && readelf -x __nv_module_id cmake_device_link.o 2>/dev/null
echo "=== core device_link.o __nv_module_id ==="
ar x $B/src/core/libninfer_core.a cmake_device_link.o 2>/dev/null && readelf -x __nv_module_id cmake_device_link.o 2>/dev/null
