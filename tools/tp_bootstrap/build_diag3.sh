#!/bin/bash
pkill -9 -f ninfer-serve 2>/dev/null; sleep 2
cp /mnt/d/Documents/workbench/ninfer/src/runtime/engine/tp2_generation_core.cpp /home/zhuojun/ninfer/src/runtime/engine/tp2_generation_core.cpp
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -3
echo "BUILD_EXIT=${PIPESTATUS[0]}"
ls -la --time-style=+%H:%M build/apps/ninfer-serve