#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
CUDALIB=/usr/local/cuda-13.1/targets/x86_64-linux/lib

echo '=== 1) minlink.cpp compiled with nvcc (no fatbin) ==='
nvcc /tmp/minlink.cpp -o /tmp/minlink_nvcc1 -lcudart -L $CUDALIB 2>/dev/null
timeout 20 /tmp/minlink_nvcc1; echo EXIT_NVCC_NOFATBIN=$?

echo '=== 2) same source as .cpp, g++, with -std=c++17 ==='
g++ -std=c++17 /tmp/minlink.cpp -o /tmp/minlink_g17 -lcudart -L $CUDALIB
timeout 20 /tmp/minlink_g17; echo EXIT_G17=$?

echo '=== 3) g++ with -O2 ==='
g++ -O2 /tmp/minlink.cpp -o /tmp/minlink_gO2 -lcudart -L $CUDALIB
timeout 20 /tmp/minlink_gO2; echo EXIT_GO2=$?

echo '=== 4) what does t1 link against? (the PASSing one) ==='
ldd /tmp/t1 | grep -E 'cudart|cuda'
echo '--- minlink_bare ldd ---'
ldd /tmp/minlink_bare | grep -E 'cudart|cuda'