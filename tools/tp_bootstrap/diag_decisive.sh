#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
CUDALIB=/usr/local/cuda-13.1/targets/x86_64-linux/lib

echo '=== 1) minlink.cpp with nvcc -arch=sm_120a (gives it a fatbin) ==='
nvcc -arch=sm_120a /tmp/minlink.cpp -o /tmp/minlink_arch -lcudart -L $CUDALIB 2>/dev/null
timeout 20 /tmp/minlink_arch; echo EXIT_MINLINK_ARCH=$?

echo '=== 2) fresh .cu: trivial kernel + cudaGetDeviceCount, nvcc -arch=sm_120a ==='
printf '__global__ void k(float* x){x[threadIdx.x]+=1.f;}\n#include <cstdio>\n#include <cuda_runtime.h>\nint main(){int n=0;cudaGetDeviceCount(&n);float*p;cudaMalloc(&p,4);k<<<1,32>>>(p);cudaDeviceSynchronize();printf("fresh count=%%d err=%%s\\n",n,cudaGetErrorName(cudaGetLastError()));return 0;}\n' > /tmp/fresh.cu
nvcc -arch=sm_120a /tmp/fresh.cu -o /tmp/fresh 2>/dev/null
timeout 20 /tmp/fresh; echo EXIT_FRESH=$?

echo '=== 3) does the tp test binary actually contain fatbins? ==='
cuobjdump ./build/tests/ninfer_tp_device_pair_test 2>/dev/null | grep -c 'Fatbin'
echo '--- target archs in tp test ---'
cuobjdump ./build/tests/ninfer_tp_device_pair_test 2>/dev/null | grep -E 'arch = ' | sort | uniq -c