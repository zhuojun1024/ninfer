#!/usr/bin/env bash
set -o pipefail
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
# core/tp
cp "$SRC/src/core/tp/weight_splitter.h" "$DST/src/core/tp/weight_splitter.h"
cp "$SRC/src/core/tp/weight_splitter.cpp" "$DST/src/core/tp/weight_splitter.cpp"
cp "$SRC/src/core/tp/tp_materialize.h" "$DST/src/core/tp/tp_materialize.h"
cp "$SRC/src/core/tp/tp_materialize.cpp" "$DST/src/core/tp/tp_materialize.cpp"
# ops/linear/fp8 (new shapes)
cp "$SRC/src/ops/linear/fp8/fp8_shapes.h" "$DST/src/ops/linear/fp8/fp8_shapes.h"
cp "$SRC/src/ops/linear/fp8/fp8_dispatch.cpp" "$DST/src/ops/linear/fp8/fp8_dispatch.cpp"
cp "$SRC/src/ops/linear/fp8/sources.cmake" "$DST/src/ops/linear/fp8/sources.cmake"
cp "$SRC/src/ops/linear/fp8/shapes/n8192_k5120.cu" "$DST/src/ops/linear/fp8/shapes/n8192_k5120.cu"
cp "$SRC/src/ops/linear/fp8/shapes/n7168_k5120.cu" "$DST/src/ops/linear/fp8/shapes/n7168_k5120.cu"
# models/qwen3_5/execution (single_layer / tp surface)
cp "$SRC/src/models/qwen3_5/execution/text.h" "$DST/src/models/qwen3_5/execution/text.h"
cp "$SRC/src/models/qwen3_5/execution/text.cpp" "$DST/src/models/qwen3_5/execution/text.cpp"
# models/qwen3_5 (TP-2 shard load path)
cp "$SRC/src/models/qwen3_5/load/tp_split_spec.h" "$DST/src/models/qwen3_5/load/tp_split_spec.h"
cp "$SRC/src/models/qwen3_5/load/tp_split_spec.cpp" "$DST/src/models/qwen3_5/load/tp_split_spec.cpp"
cp "$SRC/src/models/qwen3_5/load/tp_shard_views.h" "$DST/src/models/qwen3_5/load/tp_shard_views.h"
cp "$SRC/src/models/qwen3_5/load/tp_shard_views.cpp" "$DST/src/models/qwen3_5/load/tp_shard_views.cpp"
cp "$SRC/src/models/qwen3_5/model.h" "$DST/src/models/qwen3_5/model.h"
cp "$SRC/src/models/qwen3_5/model.cpp" "$DST/src/models/qwen3_5/model.cpp"
cp "$SRC/src/models/qwen3_5/load.h" "$DST/src/models/qwen3_5/load.h"
cp "$SRC/src/models/qwen3_5/load.cpp" "$DST/src/models/qwen3_5/load.cpp"
cp "$SRC/src/models/qwen3_5/loading_sources.cmake" "$DST/src/models/qwen3_5/loading_sources.cmake"
cp "$SRC/src/models/qwen3_5/execution/ffn.h" "$DST/src/models/qwen3_5/execution/ffn.h"
cp "$SRC/src/models/qwen3_5/execution/ffn.cpp" "$DST/src/models/qwen3_5/execution/ffn.cpp"
cp "$SRC/src/models/qwen3_5/execution/text.h" "$DST/src/models/qwen3_5/execution/text.h"
cp "$SRC/src/models/qwen3_5/execution/text.cpp" "$DST/src/models/qwen3_5/execution/text.cpp"
# tests
cp "$SRC/tests/models/qwen3_5/test_tp2_load.cpp" "$DST/tests/models/qwen3_5/test_tp2_load.cpp"
cp "$SRC/tests/models/qwen3_5/tests.cmake" "$DST/tests/models/qwen3_5/tests.cmake"
cd $DST || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DNINFER_STATIC_CUDART=ON -DBUILD_TESTING=ON > /tmp/ninfer_configure11.log 2>&1
if [ $? -ne 0 ]; then echo CONFIGURE_FAILED; tail -30 /tmp/ninfer_configure11.log; exit 1; fi
cmake --build build -j 8 > /tmp/full_build11.log 2>&1
rc=$?
echo FULL_BUILD_EXIT=$rc
[ $rc -ne 0 ] && { grep -B3 -A18 -E 'FAILED|error:' /tmp/full_build11.log | head -120; }
exit $rc