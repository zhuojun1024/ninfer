#!/usr/bin/env bash
set -o pipefail
cp /mnt/d/Documents/workbench/ninfer/src/core/tp/device_pair.cu /home/zhuojun/ninfer/src/core/tp/device_pair.cu
cp /mnt/d/Documents/workbench/ninfer/src/core/tp/device_pair.h /home/zhuojun/ninfer/src/core/tp/device_pair.h
cp /mnt/d/Documents/workbench/ninfer/tests/test_tp_device_pair.cpp /home/zhuojun/ninfer/tests/test_tp_device_pair.cpp
cp /mnt/d/Documents/workbench/ninfer/src/core/CMakeLists.txt /home/zhuojun/ninfer/src/core/CMakeLists.txt
cp /mnt/d/Documents/workbench/ninfer/tests/cmake/CoreTests.cmake /home/zhuojun/ninfer/tests/cmake/CoreTests.cmake
cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda/bin:$PATH
cmake --build build -j 8 > /tmp/ninfer_build2.log 2>&1
rc=$?
echo BUILD_EXIT=$rc
tail -6 /tmp/ninfer_build2.log
if [ $rc -ne 0 ]; then grep -B3 -A18 -E "FAILED|error:" /tmp/ninfer_build2.log | head -80; fi
exit $rc
