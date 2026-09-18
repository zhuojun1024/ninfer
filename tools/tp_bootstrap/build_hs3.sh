#!/bin/sh
set -u
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
for f in src/models/qwen3_5/execution/text.cpp; do
  cp "$SRC/$f" "$DST/$f" && echo "synced $f"
done
cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 > /tmp/ninfer_hs3.log 2>&1
rc=$?
echo BUILD_EXIT=$rc
tail -6 /tmp/ninfer_hs3.log
if [ $rc -ne 0 ]; then grep -B2 -A16 -E "FAILED|error:" /tmp/ninfer_hs3.log | head -100; fi
exit $rc
