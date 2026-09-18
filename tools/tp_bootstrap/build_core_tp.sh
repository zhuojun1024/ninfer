#!/usr/bin/env bash
set -o pipefail
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
cp "$SRC/src/core/tp/weight_splitter.h" "$DST/src/core/tp/weight_splitter.h"
cp "$SRC/src/core/tp/weight_splitter.cpp" "$DST/src/core/tp/weight_splitter.cpp"
cp "$SRC/src/core/tp/tp_materialize.h" "$DST/src/core/tp/tp_materialize.h"
cp "$SRC/src/core/tp/tp_materialize.cpp" "$DST/src/core/tp/tp_materialize.cpp"
cp "$SRC/src/models/qwen3_5/load/tp_split_spec.h" "$DST/src/models/qwen3_5/load/tp_split_spec.h"
cp "$SRC/src/models/qwen3_5/load/tp_split_spec.cpp" "$DST/src/models/qwen3_5/load/tp_split_spec.cpp"
cp "$SRC/src/core/CMakeLists.txt" "$DST/src/core/CMakeLists.txt"
cp "$SRC/src/artifact/materializer.h" "$DST/src/artifact/materializer.h"
cd $DST || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DNINFER_STATIC_CUDART=ON -DBUILD_TESTING=ON > /tmp/ninfer_configure9.log 2>&1
if [ $? -ne 0 ]; then echo CONFIGURE_FAILED; tail -30 /tmp/ninfer_configure9.log; exit 1; fi
cmake --build build --target ninfer_core -j 8 > /tmp/core_build9.log 2>&1
rc=$?
echo CORE_BUILD_EXIT=$rc
[ $rc -ne 0 ] && { grep -B3 -A18 -E 'FAILED|error:' /tmp/core_build9.log | head -100; }
exit $rc