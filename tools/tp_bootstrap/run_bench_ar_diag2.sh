#!/usr/bin/env bash
set -u
cd /home/zhuojun/ninfer
grep -rn 'CXX_STANDARD\|cxx_std' CMakeLists.txt cmake/*.cmake 2>/dev/null | head
echo '--- rebuild with c++20 ---'
/usr/local/cuda-13.1/bin/nvcc -O0 -g -std=c++20 -arch=sm_120a \
  -I/home/zhuojun/ninfer/src -I/home/zhuojun/ninfer/include \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/bench_ar_diag.cu \
  build/src/core/libninfer_core.a -lcudart -lcuda -o /home/zhuojun/prof/bench_ar_diag20 2>&1 | tail -20
echo "nvcc_status=${PIPESTATUS[0]}"
/home/zhuojun/prof/bench_ar_diag20; echo "run_exit=$?"
echo '--- gdb backtrace ---'
which gdb && gdb -batch -ex run -ex bt --args /home/zhuojun/prof/bench_ar_diag 2>&1 | tail -25
