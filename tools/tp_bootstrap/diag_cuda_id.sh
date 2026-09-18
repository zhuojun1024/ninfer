#!/usr/bin/env bash
export PATH=/usr/local/cuda/bin:$PATH
F=$(find /usr/share/cmake-3.28 -name CMakeCUDACompilerId.cpp 2>/dev/null | head -1)
echo "file=$F"
head -25 "$F"
echo "=== manual nvcc compile ==="
nvcc -x cu++ "$F" -o /tmp/cid 2>&1 | head -15
echo "NVCC_EXIT=$?"
echo "=== check g++ version used by nvcc host compiler ==="
nvcc --version | tail -2
g++ --version | head -1
echo "=== try with explicit -ccbin ==="
nvcc -x cu++ -ccbin g++-13 "$F" -o /tmp/cid2 2>&1 | head -8
echo "NVCC2_EXIT=$?"
