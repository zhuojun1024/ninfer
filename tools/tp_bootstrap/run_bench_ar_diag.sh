#!/usr/bin/env bash
set -u
cd /home/zhuojun/ninfer
/usr/local/cuda-13.1/bin/nvcc -O0 -g -std=c++17 -arch=sm_120a \
  -I/home/zhuojun/ninfer/src -I/home/zhuojun/ninfer/include \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/bench_ar_diag.cu \
  build/src/core/libninfer_core.a -lcudart -lcuda -o /home/zhuojun/prof/bench_ar_diag 2>&1 | tail -20
echo "nvcc_status=${PIPESTATUS[0]}"
ls -la /home/zhuojun/prof/bench_ar_diag
/home/zhuojun/prof/bench_ar_diag; echo "run_exit=$?"
