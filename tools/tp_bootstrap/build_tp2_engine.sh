#!/usr/bin/env bash
set -o pipefail
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
cp "$SRC/include/ninfer/types.h" "$DST/include/ninfer/types.h"
cp "$SRC/src/models/qwen3_5/frontend/resources.h" "$DST/src/models/qwen3_5/frontend/resources.h"
cp "$SRC/src/runtime/engine/tp2_generation_core.h" "$DST/src/runtime/engine/tp2_generation_core.h"
cp "$SRC/src/runtime/engine/tp2_generation_core.cpp" "$DST/src/runtime/engine/tp2_generation_core.cpp"
cp "$SRC/src/runtime/engine/engine.cpp" "$DST/src/runtime/engine/engine.cpp"
cp "$SRC/src/runtime/engine/model_instance.cpp" "$DST/src/runtime/engine/model_instance.cpp"
cp "$SRC/src/runtime/CMakeLists.txt" "$DST/src/runtime/CMakeLists.txt"
cp "$SRC/src/serve/serve_options.h" "$DST/src/serve/serve_options.h"
cp "$SRC/src/serve/serve_options.cpp" "$DST/src/serve/serve_options.cpp"
cp "$SRC/src/serve/generation_service.cpp" "$DST/src/serve/generation_service.cpp"
cd $DST || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DNINFER_STATIC_CUDART=ON -DBUILD_TESTING=ON > /tmp/ninfer_configure_tp2.log 2>&1
if [ $? -ne 0 ]; then echo CONFIGURE_FAILED; tail -40 /tmp/ninfer_configure_tp2.log; exit 1; fi
cmake --build build -j 4 > /tmp/full_build_tp2.log 2>&1
rc=$?
echo FULL_BUILD_EXIT=$rc
if [ $rc -ne 0 ]; then
  grep -nE 'FAILED|error:|Segmentation|internal compiler' /tmp/full_build_tp2.log | head -60
fi
exit $rc
