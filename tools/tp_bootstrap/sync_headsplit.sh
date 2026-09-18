#!/bin/sh
set -u
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
FILES="
src/models/qwen3_5/load/tp_split_spec.cpp
src/models/qwen3_5/load/tp_shard_views.cpp
src/models/qwen3_5/load/text.cpp
src/models/qwen3_5/execution/text.cpp
src/models/qwen3_5/execution/text.h
src/runtime/engine/tp2_generation_core.cpp
src/runtime/engine/tp2_generation_core.h
"
for f in $FILES; do
  cp "$SRC/$f" "$DST/$f" && echo "synced $f"
done
cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 > /tmp/ninfer_headsplit_build.log 2>&1
rc=$?
echo BUILD_EXIT=$rc
tail -8 /tmp/ninfer_headsplit_build.log
if [ $rc -ne 0 ]; then grep -B2 -A16 -E "FAILED|error:" /tmp/ninfer_headsplit_build.log | head -120; fi
exit $rc
