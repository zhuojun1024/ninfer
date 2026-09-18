#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
echo "=== binary timestamps ==="
ls -la --time-style=full-iso $B/apps/ninfer-serve $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | awk '{print $6, $7, $NF}'
echo ""
echo "=== does executable __nv_module_id contain CORE modules (device_pair/arena/device_cu)? ==="
readelf -x __nv_module_id $B/apps/ninfer-serve 2>/dev/null | grep -iE "device_pair|arena_cu|device_cu" | head
echo "(core module hits in serve: $(readelf -x __nv_module_id $B/apps/ninfer-serve 2>/dev/null | grep -icE 'device_pair|arena_cu|device_cu'))"
echo ""
echo "=== does executable __nv_module_id contain ops scalar module? ==="
readelf -x __nv_module_id $B/apps/ninfer-serve 2>/dev/null | grep -iE "scalar_cu" | head
echo ""
echo "=== __nv_module_ids numeric section (serve): count + duplicates ==="
readelf -x __nv_module_ids $B/apps/ninfer-serve 2>/dev/null > /tmp/serve_modids.txt
wc -l /tmp/serve_modids.txt
echo "--- raw bytes of __nv_module_ids (serve) ---"
readelf -x __nv_module_ids $B/apps/ninfer-serve 2>/dev/null | head -8
