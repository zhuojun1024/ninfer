#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /tmp
# A) exact main-build gencode form: compute_120a + sm_120a cubin, with setmaxnreg
cat > ta.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(float* x) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 32;\n" ::: "memory");
    x[threadIdx.x] += 1.0f;
}
int main() {
    float* p;
    cudaMalloc(&p, 4);
    k<<<1, 32>>>(p);
    cudaError_t e = cudaDeviceSynchronize();
    printf("A_sm120a_cubin err=%s\n", cudaGetErrorName(e));
    return e == cudaSuccess ? 0 : 1;
}
EOF
nvcc "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]" ta.cu -o ta 2>&1 | head -3
./ta; echo "A_EXIT=$?"
# B) plain sm_120 (no a) with setmaxnreg: should fail at ptxas if 'a' is required
cat > tb.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(float* x) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 32;\n" ::: "memory");
    x[threadIdx.x] += 1.0f;
}
int main() {
    float* p;
    cudaMalloc(&p, 4);
    k<<<1, 32>>>(p);
    cudaError_t e = cudaDeviceSynchronize();
    printf("B_sm120 err=%s\n", cudaGetErrorName(e));
    return e == cudaSuccess ? 0 : 1;
}
EOF
nvcc "-arch=sm_120" tb.cu -o tb 2>&1 | head -3
./tb 2>/dev/null; echo "B_EXIT=$?"
# C) compute_120a PTX only (force JIT on load), run on WSL GPU
cat > tc.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(float* x) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 32;\n" ::: "memory");
    x[threadIdx.x] += 1.0f;
}
int main() {
    float* p;
    cudaMalloc(&p, 4);
    k<<<1, 32>>>(p);
    cudaError_t e = cudaDeviceSynchronize();
    printf("C_jit err=%s\n", cudaGetErrorName(e));
    return e == cudaSuccess ? 0 : 1;
}
EOF
nvcc "-arch=compute_120a" tc.cu -o tc 2>&1 | head -3
./tc; echo "C_EXIT=$?"
