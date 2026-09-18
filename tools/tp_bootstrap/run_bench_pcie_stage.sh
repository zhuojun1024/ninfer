#!/usr/bin/env bash
set -euo pipefail
/usr/local/cuda-13.1/bin/nvcc -O2 -std=c++17 -arch=sm_120a \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/bench_pcie_stage.cu -o /home/zhuojun/prof/bench_pcie_stage
/home/zhuojun/prof/bench_pcie_stage
