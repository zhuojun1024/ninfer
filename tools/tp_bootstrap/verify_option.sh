#!/usr/bin/env bash
set -o pipefail
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
for f in CMakeLists.txt src/core/CMakeLists.txt src/serve/CMakeLists.txt src/ops/CMakeLists.txt \
  bench/context_cost/benchmarks.cmake bench/inference/benchmarks.cmake; do
  cp "$SRC/$f" "$DST/$f"
done
cd $DST || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH

echo '=== default (dynamic) configure check ==='
rm -rf build_dyn
cmake -S . -B build_dyn -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc > /tmp/ninfer_dyn.log 2>&1 \
  && echo DYN_CONFIGURE_OK || { echo DYN_CONFIGURE_FAILED; tail -20 /tmp/ninfer_dyn.log; exit 1; }

echo '=== full build with NINFER_STATIC_CUDART=ON ==='
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DNINFER_STATIC_CUDART=ON -DBUILD_TESTING=ON > /tmp/ninfer_configure5.log 2>&1
if [ $? -ne 0 ]; then echo CONFIGURE_FAILED; tail -25 /tmp/ninfer_configure5.log; exit 1; fi
cmake --build build -j 8 > /tmp/ninfer_build.log 2>&1
rc=$?
echo BUILD_EXIT=$rc
tail -4 /tmp/ninfer_build.log
[ $rc -ne 0 ] && { grep -B3 -A18 -E 'FAILED|error:' /tmp/ninfer_build.log | head -80; exit 1; }
echo '=== ldd of product CLI (expect no dynamic cudart) ==='
ldd build/apps/ninfer | grep cudart || echo '(static OK)'
echo '=== run tp test ==='
./build/tests/ninfer_tp_device_pair_test
echo TEST_EXIT=$?