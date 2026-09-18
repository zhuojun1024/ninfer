#!/usr/bin/env bash
set -o pipefail
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
for f in src/core/CMakeLists.txt src/serve/CMakeLists.txt src/ops/CMakeLists.txt \
  bench/context_cost/benchmarks.cmake bench/inference/benchmarks.cmake \
  src/core/tp/device_pair.cu src/core/tp/device_pair.h tests/test_tp_device_pair.cpp; do
  cp "$SRC/$f" "$DST/$f"
done
cd $DST || exit 1
rm -rf build
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DBUILD_TESTING=ON > /tmp/ninfer_configure4.log 2>&1
if [ $? -ne 0 ]; then echo CONFIGURE_FAILED; tail -25 /tmp/ninfer_configure4.log; exit 1; fi
cmake --build build --target ninfer_tp_device_pair_test -j 8 > /tmp/tp_test_build.log 2>&1
rc=$?
echo TEST_BUILD_EXIT=$rc
[ $rc -ne 0 ] && { grep -B3 -A15 -E 'FAILED|error:' /tmp/tp_test_build.log | head -60; exit 1; }
echo '=== ldd check (expect NO dynamic cudart) ==='
ldd build/tests/ninfer_tp_device_pair_test | grep -E 'cudart' || echo '(no dynamic cudart = static OK)'
./build/tests/ninfer_tp_device_pair_test
echo TEST_EXIT=$?