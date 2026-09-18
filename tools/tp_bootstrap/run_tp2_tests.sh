#!/bin/sh
set -u
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
for f in src/models/qwen3_5/load/text.cpp tests/models/qwen3_5/test_tp2_load.cpp tests/models/qwen3_5/test_tp2_forward.cpp; do
  cp "$SRC/$f" "$DST/$f" && echo "synced $f"
done
cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 --target ninfer_qwen3_5_tp2_load_test ninfer_qwen3_5_tp2_forward_test > /tmp/ninfer_tp2test_build.log 2>&1
rc=$?
echo BUILD_EXIT=$rc
if [ $rc -ne 0 ]; then grep -B2 -A16 -E "FAILED|error:" /tmp/ninfer_tp2test_build.log | head -80; exit 1; fi
ART=/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
echo "===== tp2_load_test ====="
./build/tests/ninfer_qwen3_5_tp2_load_test --artifact "$ART"
echo "LOAD_EXIT=$?"
echo "===== tp2_forward_test ====="
./build/tests/ninfer_qwen3_5_tp2_forward_test --artifact "$ART"
echo "FORWARD_EXIT=$?"
