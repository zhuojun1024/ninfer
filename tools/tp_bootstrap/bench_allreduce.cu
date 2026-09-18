// Standalone benchmark: measure host-staging allreduce latency on WSL2.
// Replicates DevicePair::allreduce's exact transfer pattern (4 copies + 4 syncs + CPU add)
// at the real per-layer size (hidden_size=5120 bf16 = 10240 bytes).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {     fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e));     exit(1); } } while (0)

int main(int argc, char** argv) {
    int dev_a = 0, dev_b = 1;
    int iters = 200;
    if (argc > 1) dev_a = atoi(argv[1]);
    if (argc > 2) dev_b = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);

    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { fprintf(stderr, "need 2 devices, got %d\n", ndev); return 1; }

    // Check P2P
    int can_a_to_b = 0, can_b_to_a = 0;
    cudaDeviceCanAccessPeer(&can_a_to_b, dev_a, dev_b);
    cudaDeviceCanAccessPeer(&can_b_to_a, dev_b, dev_a);
    bool p2p = (can_a_to_b && can_b_to_a);
    printf("devices: a=%d b=%d p2p=%s iters=%d\n", dev_a, dev_b, p2p ? "yes" : "no", iters);

    const std::size_t count_bytes = 5120 * 2; // 10240 bytes = hidden_size bf16
    const std::size_t elements    = count_bytes / 2;

    // Allocate on each device
    __nv_bfloat16 *da = nullptr, *db = nullptr;
    cudaSetDevice(dev_a);
    CK(cudaMalloc(&da, count_bytes));
    cudaSetDevice(dev_b);
    CK(cudaMalloc(&db, count_bytes));

    // Pinned staging buffer [a | b]
    __nv_bfloat16* pinned = nullptr;
    CK(cudaHostAlloc(&pinned, 2 * count_bytes, cudaHostAllocDefault));

    // Streams (independent, non-blocking — matches DevicePair)
    cudaStream_t sa, sb;
    cudaSetDevice(dev_a);
    CK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
    cudaSetDevice(dev_b);
    CK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        cudaSetDevice(dev_a);
        CK(cudaMemcpyAsync(pinned, da, count_bytes, cudaMemcpyDeviceToHost, sa));
        cudaSetDevice(dev_b);
        CK(cudaMemcpyAsync(pinned + elements, db, count_bytes, cudaMemcpyDeviceToHost, sb));
        cudaSetDevice(dev_a);
        CK(cudaStreamSynchronize(sa));
        cudaSetDevice(dev_b);
        CK(cudaStreamSynchronize(sb));
        for (std::size_t j = 0; j < elements; ++j)
            pinned[j] = __hadd(pinned[j], pinned[elements + j]);
        cudaSetDevice(dev_a);
        CK(cudaMemcpyAsync(da, pinned, count_bytes, cudaMemcpyHostToDevice, sa));
        cudaSetDevice(dev_b);
        CK(cudaMemcpyAsync(db, pinned, count_bytes, cudaMemcpyHostToDevice, sb));
        cudaSetDevice(dev_a);
        CK(cudaStreamSynchronize(sa));
        cudaSetDevice(dev_b);
        CK(cudaStreamSynchronize(sb));
    }

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cudaSetDevice(dev_a);
        CK(cudaMemcpyAsync(pinned, da, count_bytes, cudaMemcpyDeviceToHost, sa));
        cudaSetDevice(dev_b);
        CK(cudaMemcpyAsync(pinned + elements, db, count_bytes, cudaMemcpyDeviceToHost, sb));
        cudaSetDevice(dev_a);
        CK(cudaStreamSynchronize(sa));
        cudaSetDevice(dev_b);
        CK(cudaStreamSynchronize(sb));
        for (std::size_t j = 0; j < elements; ++j)
            pinned[j] = __hadd(pinned[j], pinned[elements + j]);
        cudaSetDevice(dev_a);
        CK(cudaMemcpyAsync(da, pinned, count_bytes, cudaMemcpyHostToDevice, sa));
        cudaSetDevice(dev_b);
        CK(cudaMemcpyAsync(db, pinned, count_bytes, cudaMemcpyHostToDevice, sb));
        cudaSetDevice(dev_a);
        CK(cudaStreamSynchronize(sa));
        cudaSetDevice(dev_b);
        CK(cudaStreamSynchronize(sb));
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_op_ms = total_ms / iters;
    double per_token_ms = per_op_ms * 64; // 64 layers
    double implied_toks = 1000.0 / per_token_ms;

    printf("allreduce: %.3f ms/op (%zu bytes)\n", per_op_ms, count_bytes);
    printf("64 layers: %.1f ms/token\n", per_token_ms);
    printf("implied decode: %.1f tok/s (allreduce only)\n", implied_toks);

    // Also measure a single D2H+sync round-trip for reference
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cudaSetDevice(dev_a);
        CK(cudaMemcpyAsync(pinned, da, count_bytes, cudaMemcpyDeviceToHost, sa));
        CK(cudaStreamSynchronize(sa));
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double d2h_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;
    printf("single D2H+sync: %.3f ms (%zu bytes)\n", d2h_ms, count_bytes);

    cudaSetDevice(dev_a);
    cudaFree(da);
    cudaSetDevice(dev_b);
    cudaFree(db);
    cudaFreeHost(pinned);
    return 0;
}
