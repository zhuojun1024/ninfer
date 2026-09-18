#!/bin/sh
set -u
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
for f in src/core/tp/device_pair.h src/core/tp/device_pair.cu src/models/qwen3_5/execution/text.h tests/test_tp_device_pair.cpp; do
  cp "$SRC/$f" "$DST/$f" && echo "synced $f"
done
echo "---"
grep -c stream_a "$DST/src/core/tp/device_pair.cu"
