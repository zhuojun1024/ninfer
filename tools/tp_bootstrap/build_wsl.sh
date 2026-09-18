#!/usr/bin/env bash
set -o pipefail
cd /home/zhuojun/ninfer || exit 1
rm -rf build
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc > /tmp/ninfer_configure.log 2>&1
if [ $? -ne 0 ]; then echo CONFIGURE_FAILED; tail -25 /tmp/ninfer_configure.log; exit 1; fi
cmake --build build -j 8 > /tmp/ninfer_build.log 2>&1
rc=$?
echo BUILD_EXIT=$rc
tail -6 /tmp/ninfer_build.log
if [ $rc -ne 0 ]; then grep -B3 -A18 -E "FAILED|error:" /tmp/ninfer_build.log | head -80; fi
exit $rc
