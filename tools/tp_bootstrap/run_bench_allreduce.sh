#!/usr/bin/env bash
set -euo pipefail
/usr/local/cuda-13.1/bin/nvcc -O2 -std=c++17 -arch=sm_120a /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/bench_allreduce.cu -o /tmp/bench_allreduce
/tmp/bench_allreduce 0 1 300
