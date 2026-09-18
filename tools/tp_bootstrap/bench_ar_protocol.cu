// Two-GPU allreduce protocol bench: does slice-pipelining recover the PCIe full-duplex half?
//
// The engine's TP-2 allreduce over host staging does, per call: write the local delta to mapped
// pinned host memory, fence, publish an arrival token, spin for the peer's token, then read the
// peer's staging. The write and read phases are serial, so a full-duplex link is busy only half the
// time. This bench reproduces that protocol on both RTX 5060 Ti cards and compares it against a
// sliced variant that orders the writes (slice k only after slice k-1 landed) so the peer can read
// early slices while later ones are still going out.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void ar_protocol(uint4* __restrict__ mine_dev, uint4* __restrict__ mine_host,
                            uint4* __restrict__ peer_host, int* order, int* arrival_mine,
                            const int* arrival_peer, int units, int slice_units, int token,
                            int pipeline) {
    const int b = blockIdx.x;
    if (pipeline != 0) {
        if (threadIdx.x == 0 && b > 0) {
            while (*(volatile int*)&order[b - 1] != token) { __nanosleep(64); }
        }
        __syncthreads();
    }
    const int lo = b * slice_units;
    const int hi = min(lo + slice_units, units);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) { mine_host[i] = mine_dev[i]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile int*)&order[b]       = token;
        *(volatile int*)&arrival_mine[b] = token;
    }
    if (threadIdx.x == 0) {
        while (*(const volatile int*)&arrival_peer[b] != token) { __nanosleep(64); }
    }
    __syncthreads();
    __threadfence_system();
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const uint4 p = peer_host[i];
        uint4       m = mine_dev[i];
        m.x += p.x; m.y += p.y; m.z += p.z; m.w += p.w;
        mine_dev[i] = m;
    }
}

struct Buf {
    uint4* dev = nullptr;
    uint4* host = nullptr;
    int*   order = nullptr;
    int*   arrival = nullptr;
    int    device = 0;
};

static Buf make_buf(int device, std::size_t bytes) {
    Buf b;
    b.device = device;
    cudaSetDevice(device);
    if (cudaMalloc(&b.dev, bytes) != cudaSuccess) { printf("cudaMalloc failed\n"); exit(1); }
    void* h = nullptr;
    if (cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        printf("cudaHostAlloc failed\n"); exit(1);
    }
    b.host = (uint4*)h;
    void* o = nullptr;
    cudaHostAlloc(&o, 4096, cudaHostAllocMapped | cudaHostAllocPortable);
    b.order = (int*)o;
    void* a = nullptr;
    cudaHostAlloc(&a, 4096, cudaHostAllocMapped | cudaHostAllocPortable);
    b.arrival = (int*)a;
    cudaMemset(b.dev, 1, bytes);
    cudaMemset(b.host, 2, bytes);
    return b;
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { printf("cudaGetDeviceCount failed\n"); return 1; }
    printf("devices=%d\n", count);
    const std::size_t sizes[] = {5120u * 1024u * 2u, 5120u * 189u * 2u, 5120u * 44u * 2u, 5120u * 2u};
    const int         slices[] = {1, 2, 4, 8, 16, 32};
    cudaStream_t sa, sb;
    cudaSetDevice(0); cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking);
    cudaSetDevice(1); cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking);

    cudaEvent_t a0, a1, b0, b1;
    cudaSetDevice(0);
    cudaEventCreate(&a0); cudaEventCreate(&a1);
    cudaSetDevice(1);
    cudaEventCreate(&b0); cudaEventCreate(&b1);

    for (std::size_t bytes : sizes) {
        Buf A = make_buf(0, bytes), B = make_buf(1, bytes);
        const int units = (int)(bytes / 16);
        int       token = 1000;
        printf("\npayload=%zu B (T=%zu)\n", bytes, bytes / (5120 * 2));
        for (int k : slices) {
            const int  slice_units = (units + k - 1) / k;
            const bool pipeline    = k > 1;
            double     best        = 1e9;
            const int  iters       = bytes > (1u << 21) ? 30 : 300;
            for (int it = 0; it < iters; ++it) {
                ++token; // strictly increasing: stale slot values never alias the expected token
                cudaSetDevice(0);
                cudaEventRecord(a0, sa);
                ar_protocol<<<k, 1024, 0, sa>>>(A.dev, A.host, B.host, A.order, A.arrival, B.arrival,
                                                units, slice_units, token, pipeline);
                cudaEventRecord(a1, sa);
                cudaSetDevice(1);
                cudaEventRecord(b0, sb);
                ar_protocol<<<k, 1024, 0, sb>>>(B.dev, B.host, A.host, B.order, B.arrival, A.arrival,
                                                units, slice_units, token, pipeline);
                cudaEventRecord(b1, sb);
                cudaSetDevice(0);
                cudaEventSynchronize(a1);
                cudaSetDevice(1);
                cudaEventSynchronize(b1);
                float msa = 0.0f, msb = 0.0f;
                cudaSetDevice(0); cudaEventElapsedTime(&msa, a0, a1);
                cudaSetDevice(1); cudaEventElapsedTime(&msb, b0, b1);
                const double ms = msa > msb ? msa : msb;
                if (ms < best) { best = ms; }
            }
            printf("  slices=%2d %s  best %8.3f ms  (%6.2f GB/s round trip)\n", k,
                   pipeline ? "pipelined" : "serial   ", best, 2.0 * bytes / (best * 1e-3) / 1e9);
        }
        cudaSetDevice(0); cudaFree(A.dev); cudaFreeHost(A.host); cudaFreeHost(A.order); cudaFreeHost(A.arrival);
        cudaSetDevice(1); cudaFree(B.dev); cudaFreeHost(B.host); cudaFreeHost(B.order); cudaFreeHost(B.arrival);
    }
    return 0;
}
