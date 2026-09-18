#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
echo "=== SERVE: fatbin image kinds ==="
cuobjdump $B/apps/ninfer-serve 2>/dev/null | grep -E "Reloc|SASS|PTX|sm_" | sort | uniq -c | head -20
echo "=== TEST: fatbin image kinds ==="
cuobjdump $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -E "Reloc|SASS|PTX|sm_" | sort | uniq -c | head -20
echo "=== SERVE: set_i32_scalar_kernel in fatbin (any form) ==="
cuobjdump $B/apps/ninfer-serve 2>/dev/null | grep -c "set_i32_scalar_kernel"
echo "=== TEST: set_i32_scalar_kernel in fatbin ==="
cuobjdump $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -c "set_i32_scalar_kernel"
echo "=== libninfer_ops.a: device_link.o SASS check ==="
cd /tmp && rm -rf dlinkcheck && mkdir dlinkcheck && cd dlinkcheck
ar x $B/src/ops/libninfer_ops.a cmake_device_link.o 2>/dev/null && cuobjdump cmake_device_link.o 2>/dev/null | grep -E "Reloc|SASS|sm_" | sort | uniq -c | head
echo "=== ops archive: does scalar.cu.o fatbin have SASS or relocatable? ==="
ar x $B/src/ops/libninfer_ops.a scalar.cu.o 2>/dev/null && cuobjdump scalar.cu.o 2>/dev/null | grep -E "Reloc|SASS|sm_" | sort | uniq -c | head
