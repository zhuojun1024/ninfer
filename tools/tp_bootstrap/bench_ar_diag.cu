#include "core/tp/device_pair.h"
#include <cuda_runtime.h>
#include <cstdio>
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("start\n");
    { ninfer::DeviceContext d0(0), d1(1); printf("ctx ok %p %p\n", (void*)d0.stream, (void*)d1.stream); }
    printf("before pair\n");
    {
        ninfer::tp::DevicePair pair(0, 1);
        printf("pair ok p2p=%d streamsa=%p b=%p\n", pair.p2p_available() ? 1 : 0,
               (void*)pair.a().stream, (void*)pair.b().stream);
        void* pa = nullptr; void* pb = nullptr;
        cudaSetDevice(0); cudaMalloc(&pa, 10240);
        cudaSetDevice(1); cudaMalloc(&pb, 10240);
        printf("buffers ok\n");
        for (int i = 0; i < 3; ++i) {
            pair.allreduce(pa, pb, 10240, pair.a().stream, pair.b().stream);
            cudaSetDevice(0); cudaStreamSynchronize(pair.a().stream);
            cudaSetDevice(1); cudaStreamSynchronize(pair.b().stream);
        }
        printf("small AR ok\n");
        void* qa = nullptr; void* qb = nullptr;
        const std::size_t big = 5120ULL * 1024 * 2;
        cudaSetDevice(0); cudaMalloc(&qa, big);
        cudaSetDevice(1); cudaMalloc(&qb, big);
        for (int i = 0; i < 3; ++i) {
            pair.allreduce(qa, qb, big, pair.a().stream, pair.b().stream);
            cudaSetDevice(0); printf("  sync a %d\n", i); cudaStreamSynchronize(pair.a().stream);
            cudaSetDevice(1); printf("  sync b %d\n", i); cudaStreamSynchronize(pair.b().stream);
        }
        printf("big AR ok\n");
    }
    printf("done\n");
    return 0;
}
