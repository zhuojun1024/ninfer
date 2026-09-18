#!/bin/bash
cd /home/zhuojun/ninfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
grep -n 'cstdio' src/runtime/engine/tp2_generation_core.cpp | head -3
cmake --build build -j 8 2>&1 | grep -B3 -A8 'error' | head -50