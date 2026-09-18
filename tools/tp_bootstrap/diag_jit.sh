#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /tmp
# 1) plain 13.1 runtime, no ninfer code
cat > t1.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    printf("count=%d err=%s\n", n, cudaGetErrorName(e));
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, i);
        printf("dev%d %s cc=%d.%d\n", i, p.name, p.major, p.minor);
    }
    return 0;
}
EOF
nvcc -arch=sm_120a t1.cu -o t1 && ./t1
echo "T1_EXIT=$?"
# 2) arch-specific kernel (setmaxnreg is sm_90a+/sm_120a)
cat > t2.cu <<'EOF'
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
    printf("arch_kernel err=%s\n", cudaGetErrorName(e));
    return e == cudaSuccess ? 0 : 1;
}
EOF
nvcc -arch=sm_120a t2.cu -o t2 && ./t2
echo "T2_EXIT=$?"
# 3) mxf4nvf4 mma (sm_120a only)
cat > t3.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(float* d) {
    unsigned a0=1,a1=1,a2=1,a3=1, b0=1,b1=1, c0=0,c1=0,c2=0,c3=0, sfa=1, sfb=1;
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.f32.e8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, %15;\n"
      : "=f"(c0),"=f"(c1),"=f"(c2),"=f"(c3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
        "f"(c0),"f"(c1),"f"(c2),"f"(c3), "r"(sfa), "r"(sfb));
    d[threadIdx.x] = c0;
}
int main() {
    float* p;
    cudaMalloc(&p, 4);
    k<<<1, 32>>>(p);
    cudaError_t e = cudaDeviceSynchronize();
    printf("mxf4nvf4 err=%s\n", cudaGetErrorName(e));
    return e == cudaSuccess ? 0 : 1;
}
EOF
nvcc -arch=sm_120a t3.cu -o t3 2>&1 | head -5 && ./t3
echo "T3_EXIT=$?"
