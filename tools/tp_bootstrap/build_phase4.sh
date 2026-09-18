#!/usr/bin/env bash
set -o pipefail
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
cp "$SRC/src/core/tp/tp_materialize.cpp" "$DST/src/core/tp/tp_materialize.cpp"
cp "$SRC/src/core/tp/tp_materialize.h" "$DST/src/core/tp/tp_materialize.h"
cp "$SRC/src/core/tp/weight_splitter.h" "$DST/src/core/tp/weight_splitter.h"
cp "$SRC/src/core/tp/weight_splitter.cpp" "$DST/src/core/tp/weight_splitter.cpp"
cp "$SRC/src/models/qwen3_5/execution/text.h" "$DST/src/models/qwen3_5/execution/text.h"
cp "$SRC/src/models/qwen3_5/execution/text.cpp" "$DST/src/models/qwen3_5/execution/text.cpp"
cp "$SRC/src/models/qwen3_5/execution/ffn.cpp" "$DST/src/models/qwen3_5/execution/ffn.cpp"
cp "$SRC/src/models/qwen3_5/load/tp_shard_views.cpp" "$DST/src/models/qwen3_5/load/tp_shard_views.cpp"
cp "$SRC/src/ops/linear/fp8/fp8_shapes.h" "$DST/src/ops/linear/fp8/fp8_shapes.h"
cp "$SRC/src/ops/linear/fp8/fp8_dispatch.cpp" "$DST/src/ops/linear/fp8/fp8_dispatch.cpp"
cp "$SRC/src/ops/linear/fp8/sources.cmake" "$DST/src/ops/linear/fp8/sources.cmake"
cp "$SRC/src/ops/linear/fp8/shapes/n124160_k5120.cu" "$DST/src/ops/linear/fp8/shapes/n124160_k5120.cu"
cp "$SRC/src/ops/linear/fp8/shapes/n17408_k5120.cu" "$DST/src/ops/linear/fp8/shapes/n17408_k5120.cu"
cp "$SRC/src/ops/linear/fp8/shapes/n5120_k8704.cu" "$DST/src/ops/linear/fp8/shapes/n5120_k8704.cu"
cp "$SRC/src/ops/softmax_attention/dense/causal_cache/prompt.cu" "$DST/src/ops/softmax_attention/dense/causal_cache/prompt.cu"
cp "$SRC/tests/models/qwen3_5/test_tp2_forward.cpp" "$DST/tests/models/qwen3_5/test_tp2_forward.cpp"
cp "$SRC/tests/models/qwen3_5/tests.cmake" "$DST/tests/models/qwen3_5/tests.cmake"
cd $DST || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DNINFER_STATIC_CUDART=ON -DBUILD_TESTING=ON > /tmp/ninfer_configure15.log 2>&1
if [ $? -ne 0 ]; then echo CONFIGURE_FAILED; tail -30 /tmp/ninfer_configure15.log; exit 1; fi
cmake --build build -j 4 > /tmp/full_build18.log 2>&1
rc=$?
echo FULL_BUILD_EXIT=$rc
if [ $rc -ne 0 ]; then
  grep -nE 'FAILED|error|Segmentation|internal compiler' /tmp/full_build18.log | head -40
fi
exit $rc
