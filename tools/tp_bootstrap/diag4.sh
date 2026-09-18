#!/usr/bin/env bash
export PATH=/usr/local/cuda/bin:$PATH
echo "=== which nvcc after export ==="
which nvcc
nvcc --version | tail -1
echo "=== cache ==="
grep -i "CUDA_COMPILER" /home/zhuojun/ninfer/build/CMakeCache.txt 2>/dev/null
echo "=== date in WSL ==="
date
echo "=== /usr/local/cuda target ==="
ls -la /usr/local/ | grep cuda
