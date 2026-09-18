#!/bin/bash
pkill -9 -f ninfer-serve 2>/dev/null; sleep 2
for f in src/runtime/engine/tp2_generation_core.h src/runtime/engine/tp2_generation_core.cpp; do
  cp /mnt/d/Documents/workbench/ninfer/$f /home/zhuojun/ninfer/$f
done
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -4
echo "BUILD_EXIT=${PIPESTATUS[0]}"
ls -la --time-style=+%H:%M build/apps/ninfer-serve