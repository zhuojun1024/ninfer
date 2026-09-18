#!/usr/bin/env bash
set -u
cd /home/zhuojun/ninfer
ls build/src/core/*.a
/usr/local/cuda-13.1/bin/nvcc -O2 -std=c++17 -arch=sm_120a \
  -I/home/zhuojun/ninfer/src -I/home/zhuojun/ninfer/include \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/bench_ar_tp2.cu \
  build/src/core/libninfer_core.a -lcudart -lcuda -o /home/zhuojun/prof/bench_ar_tp2 2>&1 | tail -30
echo "build_exit=$?"
/home/zhuojun/prof/bench_ar_tp2 0 1 20
