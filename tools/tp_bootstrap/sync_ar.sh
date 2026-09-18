#!/bin/sh
set -u
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
for f in src/core/tp/device_pair.h src/core/tp/device_pair.cu; do
  cp "$SRC/$f" "$DST/$f" && echo "synced $f"
done
echo "---"
