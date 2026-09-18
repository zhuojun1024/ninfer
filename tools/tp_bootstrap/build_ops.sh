#!/usr/bin/env bash
set -o pipefail
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
cp "$SRC/src/ops/linear/fp8/shapes/n8192_k5120.cu" "$DST/src/ops/linear/fp8/shapes/n8192_k5120.cu"
cp "$SRC/src/ops/linear/fp8/shapes/n7168_k5120.cu" "$DST/src/ops/linear/fp8/shapes/n7168_k5120.cu"
cp "$SRC/src/ops/linear/fp8/fp8_shapes.h" "$DST/src/ops/linear/fp8/fp8_shapes.h"
cp "$SRC/src/ops/linear/fp8/fp8_dispatch.cpp" "$DST/src/ops/linear/fp8/fp8_dispatch.cpp"
cp "$SRC/src/ops/linear/fp8/sources.cmake" "$DST/src/ops/linear/fp8/sources.cmake"
cd $DST || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DNINFER_STATIC_CUDART=ON -DBUILD_TESTING=ON > /tmp/ninfer_configure10.log 2>&1
if [ $? -ne 0 ]; then echo CONFIGURE_FAILED; tail -30 /tmp/ninfer_configure10.log; exit 1; fi
cmake --build build --target ninfer_ops -j 8 > /tmp/ops_build10.log 2>&1
rc=$?
echo OPS_BUILD_EXIT=$rc
[ $rc -ne 0 ] && { grep -B3 -A18 -E 'FAILED|error:' /tmp/ops_build10.log | head -120; }
exit $rc