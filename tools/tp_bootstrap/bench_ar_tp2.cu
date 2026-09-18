// Standalone benchmark for the TP-2 in-kernel allreduce (WSL2, no P2P): per-call latency and
// achieved PCIe bandwidth against payload size, using the production DevicePair path.
#include "core/tp/device_pair.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                          \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,    \
                    cudaGetErrorString(e_));                                          \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    int dev_a = 0;
    int dev_b = 1;
    int iters = 20;
    if (argc > 1) { dev_a = std::atoi(argv[1]); }
    if (argc > 2) { dev_b = std::atoi(argv[2]); }
    if (argc > 3) { iters = std::atoi(argv[3]); }

    ninfer::tp::DevicePair pair(dev_a, dev_b);
    const int hidden = 5120;
    printf("devices a=%d b=%d iters=%d p2p=%d\n", dev_a, dev_b, iters,
           pair.p2p_available() ? 1 : 0);

    const std::vector<std::size_t> sizes = {10ULL * 1024, 65536, 640ULL * 1024,
                                            5120ULL * 189 * 2, 5120ULL * 512 * 2,
                                            5120ULL * 1024 * 2};
    for (const std::size_t bytes : sizes) {
        void* pa = nullptr;
        void* pb = nullptr;
        cudaSetDevice(dev_a);
        CK(cudaMalloc(&pa, bytes));
        CK(cudaMemset(pa, 0, bytes));
        cudaSetDevice(dev_b);
        CK(cudaMalloc(&pb, bytes));
        CK(cudaMemset(pb, 0, bytes));
        const std::size_t elements = bytes / 2;
        printf("payload %8zu bytes (T=%5zu): ", bytes, elements / hidden);

        // Pipelined: queue every call back to back on the two compute streams, like the layer loop.
        std::vector<double> per_call;
        for (int round = 0; round < 3; ++round) {
            pair.allreduce(pa, pb, bytes, pair.a().stream, pair.b().stream);
        }
        cudaSetDevice(dev_a);
        CK(cudaStreamSynchronize(pair.a().stream));
        cudaSetDevice(dev_b);
        CK(cudaStreamSynchronize(pair.b().stream));

        const Clock::time_point t0 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            pair.allreduce(pa, pb, bytes, pair.a().stream, pair.b().stream);
        }
        cudaSetDevice(dev_a);
        CK(cudaStreamSynchronize(pair.a().stream));
        cudaSetDevice(dev_b);
        CK(cudaStreamSynchronize(pair.b().stream));
        const double pipelined_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;

        // Serialized: one call per sync, to expose the peer round-trip latency.
        std::vector<double> serial;
        for (int i = 0; i < std::min(iters, 10); ++i) {
            const Clock::time_point s = Clock::now();
            pair.allreduce(pa, pb, bytes, pair.a().stream, pair.b().stream);
            cudaSetDevice(dev_a);
            CK(cudaStreamSynchronize(pair.a().stream));
            cudaSetDevice(dev_b);
            CK(cudaStreamSynchronize(pair.b().stream));
            serial.push_back(std::chrono::duration<double, std::milli>(Clock::now() - s).count());
        }
        std::sort(serial.begin(), serial.end());
        const double serial_ms = serial[serial.size() / 2];
        // Each device moves 'bytes' to host and reads 'bytes' back over its own PCIe link.
        printf("pipelined %7.3f ms/call  serial %7.3f ms  link %6.2f GB/s\n", pipelined_ms,
               serial_ms, 2.0 * static_cast<double>(bytes) / (serial_ms * 1e6) * 1e3);
        cudaSetDevice(dev_a);
        CK(cudaFree(pa));
        cudaSetDevice(dev_b);
        CK(cudaFree(pb));
    }
    return 0;
}
