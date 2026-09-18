#include <cuda_runtime.h>
#include <cstdio>
int main(int argc, char** argv) {
    int dev = (argc > 1) ? atoi(argv[1]) : 0;
    cudaSetDevice(dev);
    size_t gb = 14ULL << 30;
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, gb);
    if (e == cudaSuccess) {
        printf("dev %d: allocated %zu GB OK (stale accounting, memory is free)\n", dev, gb >> 30);
        cudaFree(p);
    } else {
        printf("dev %d: cudaMalloc %zu GB FAILED: %s (memory really pinned)\n", dev, gb >> 30, cudaGetErrorString(e));
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("dev %d total VRAM: %zu MB\n", dev, prop.totalGlobalMem >> 20);
    return 0;
}
