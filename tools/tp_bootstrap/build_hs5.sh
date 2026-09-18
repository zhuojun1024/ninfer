#!/bin/bash
set -e
cd /mnt/d/Documents/workbench/ninfer
cp src/ops/launcher/causal_conv1d.h /home/zhuojun/ninfer/src/ops/launcher/causal_conv1d.h
cp src/ops/launcher/causal_conv1d.cu /home/zhuojun/ninfer/src/ops/launcher/causal_conv1d.cu
cp src/ops/wrapper/causal_conv1d_silu.cpp /home/zhuojun/ninfer/src/ops/wrapper/causal_conv1d_silu.cpp
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -8
echo "BUILD_EXIT=${PIPESTATUS[0]}"
