#!/bin/bash
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build --target ninfer-serve -j 8 2>&1 | tail -5
echo "BUILD_EXIT=${PIPESTATUS[0]}"
ls -la --time-style=+%H:%M build/ninfer-serve
