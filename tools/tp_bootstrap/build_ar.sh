#!/bin/sh
set -u
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
for f in src/core/tp/device_pair.h src/core/tp/device_pair.cu; do
  cp "$SRC/$f" "$DST/$f" && echo "synced $f"
done
cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -40
