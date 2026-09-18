#!/usr/bin/env bash
set -o pipefail
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
cp "$SRC/src/core/tp/weight_splitter.h" "$DST/src/core/tp/weight_splitter.h"
cp "$SRC/src/core/tp/weight_splitter.cpp" "$DST/src/core/tp/weight_splitter.cpp"
cp "$SRC/src/core/CMakeLists.txt" "$DST/src/core/CMakeLists.txt"
cp "$SRC/tests/ops/linear/test_tp2_split_nvfp4.cpp" "$DST/tests/ops/linear/test_tp2_split_nvfp4.cpp"
cd $DST || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DNINFER_STATIC_CUDART=ON -DBUILD_TESTING=ON > /tmp/ninfer_configure8.log 2>&1
if [ $? -ne 0 ]; then echo CONFIGURE_FAILED; tail -25 /tmp/ninfer_configure8.log; exit 1; fi
cmake --build build --target ninfer_linear_tp2_split_nvfp4_test -j 8 > /tmp/tp2_test_build8.log 2>&1
rc=$?
echo TEST_BUILD_EXIT=$rc
[ $rc -ne 0 ] && { grep -B3 -A18 -E 'FAILED|error:' /tmp/tp2_test_build8.log | head -80; }
exit $rc