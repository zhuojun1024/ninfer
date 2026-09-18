// Measure the PCIe host-staging bandwidth the TP-2 allreduce rides on: a GPU writes its partial
// sum into mapped pinned host memory, then reads the peer's buffer back. Reports both directions
// at the decode payload (hidden_size bf16 = 10 KiB) and the T=256 prefill payload (2.5 MiB).
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    const std::size_t sizes[] = {5120 * 2, 5120 * 256 * 2};
    for (const std::size_t bytes : sizes) {
        void* dev = nullptr;
        void* host = nullptr;
        if (cudaMalloc(&dev, bytes) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }
        if (cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) !=
            cudaSuccess) {
            printf("cudaHostAlloc failed\n");
            return 1;
        }
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaMemset(dev, 1, bytes);
        for (int i = 0; i < 10; ++i) {
            cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        const int iters = bytes < (1u << 16) ? 2000 : 200;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        const double d2h = std::chrono::duration<double>(t1 - t0).count() / iters;
        const double h2d = std::chrono::duration<double>(t2 - t1).count() / iters;
        printf("payload=%7zu B  D2H %8.1f us (%5.2f GB/s)  H2D %8.1f us (%5.2f GB/s)  "
               "one AR (write+read) %8.1f us -> %5.2f GB/s\n",
               bytes, d2h * 1e6, bytes / d2h / 1e9, h2d * 1e6, bytes / h2d / 1e9, (d2h + h2d) * 1e6,
               2.0 * bytes / (d2h + h2d) / 1e9);
        cudaFreeHost(host);
        cudaFree(dev);
    }
    printf("x8 Gen4 theoretical one-way: 16 GB/s\n");
    return 0;
}
