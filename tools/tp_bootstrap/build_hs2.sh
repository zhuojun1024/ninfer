#!/bin/sh
set -u
cp /mnt/d/Documents/workbench/ninfer/src/ops/weight_input.cpp /home/zhuojun/ninfer/src/ops/weight_input.cpp
echo synced weight_input.cpp
cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 > /tmp/ninfer_hs2.log 2>&1
rc=$?
echo BUILD_EXIT=$rc
tail -6 /tmp/ninfer_hs2.log
if [ $rc -ne 0 ]; then grep -B2 -A16 -E "FAILED|error:" /tmp/ninfer_hs2.log | head -100; fi
exit $rc
