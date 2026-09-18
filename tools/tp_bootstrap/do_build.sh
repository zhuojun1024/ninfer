#!/usr/bin/env bash
cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 4 > /tmp/full_build_tp2.log 2>&1
rc=$?
echo "BUILD_RC=$rc" >> /tmp/full_build_tp2.log
if [ $rc -ne 0 ]; then
  grep -nE 'FAILED|error:' /tmp/full_build_tp2.log | head -40 >> /tmp/full_build_tp2.log
fi
