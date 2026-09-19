// TP-2 mapped-pinned spike: verifies the in-kernel allreduce primitive that NInfer's TP-2 path needs
// when peer access is unavailable (no NVLink/P2P, SYS topology) and reports its latency and bandwidth
// outside the engine. The engine uses the same pair of mapped pinned buffers plus arrival tokens.
//
// Build and run (a vcvars64 environment must be active):
//   nvcc -arch=sm_120a -O2 -o build-win/tp_mapped_spike.exe tools/win_port/tp_mapped_spike.cu
//   build-win/tp_mapped_spike.exe [device_a] [device_b]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(expr)                                                                                \
    do {                                                                                           \
        const cudaError_t status_ = (expr);                                                        \
        if (status_ != cudaSuccess) {                                                              \
            std::printf("FAIL %s:%d  %s -> %s\n", __FILE__, __LINE__, #expr,                      \
                        cudaGetErrorString(status_));                                              \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

__global__ void fill_kernel(std::uint8_t* destination, std::size_t bytes, unsigned seed) {
    const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < bytes) { destination[index] = static_cast<std::uint8_t>(seed + (index & 0xFFU)); }
}

__global__ void sum_kernel(const std::uint8_t* source, std::size_t bytes,
                           unsigned long long* total) {
    const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < bytes) { atomicAdd(total, static_cast<unsigned long long>(source[index])); }
}

float milliseconds(cudaEvent_t start, cudaEvent_t stop) {
    float value = 0.0F;
    CHECK(cudaEventElapsedTime(&value, start, stop));
    return value;
}

} // namespace

int main(int argc, char** argv) {
    const int device_a = argc > 1 ? std::atoi(argv[1]) : 0;
    const int device_b = argc > 2 ? std::atoi(argv[2]) : 1;

    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    std::printf("devices=%d  pair=(%d,%d)\n", count, device_a, device_b);
    if (count < 2) { std::printf("FAIL: two devices are required\n"); return 1; }

    for (const int device : {device_a, device_b}) {
        cudaDeviceProp properties{};
        CHECK(cudaGetDeviceProperties(&properties, device));
        std::printf("  device %d: %s  sm_%d%d  %.1f GiB  unified=%d\n", device, properties.name,
                    properties.major, properties.minor,
                    static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                    properties.unifiedAddressing);
    }

    int can_a_to_b = 0;
    int can_b_to_a = 0;
    CHECK(cudaDeviceCanAccessPeer(&can_a_to_b, device_a, device_b));
    CHECK(cudaDeviceCanAccessPeer(&can_b_to_a, device_b, device_a));
    std::printf("peer access: %d->%d %d, %d->%d %d\n", device_a, device_b, can_a_to_b, device_b,
                device_a, can_b_to_a);

    constexpr std::size_t kStagingBytes = 24ULL << 20; // matches kInKernelArBytes in device_pair.cu
    void* host_a = nullptr;
    void* host_b = nullptr;
    void* device_a_view = nullptr;
    void* device_b_view = nullptr;

    CHECK(cudaSetDevice(device_a));
    CHECK(cudaHostAlloc(&host_a, kStagingBytes, cudaHostAllocPortable | cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer(&device_a_view, host_a, 0));
    CHECK(cudaSetDevice(device_b));
    CHECK(cudaHostAlloc(&host_b, kStagingBytes, cudaHostAllocPortable | cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer(&device_b_view, host_b, 0));
    std::printf("mapped pinned: host_a=%p view_a=%p | host_b=%p view_b=%p\n", host_a, device_a_view,
                host_b, device_b_view);

    // Cross-device reachability: the device pointer of one allocation must be usable by the peer.
    void* peer_view = nullptr;
    CHECK(cudaSetDevice(device_b));
    CHECK(cudaHostGetDevicePointer(&peer_view, host_a, 0));
    std::printf("peer view of host_a from device %d: %p\n", device_b, peer_view);

    unsigned long long* sum = nullptr;
    CHECK(cudaMalloc(&sum, sizeof(unsigned long long)));
    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    CHECK(cudaStreamCreate(&stream_a));
    CHECK(cudaStreamCreate(&stream_b));
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Establish the mapping of the pinned pages before timing: the first device access to freshly
    // allocated mapped memory pays a page-mapping cost that would otherwise show up as a slow 4 KiB row.
    CHECK(cudaSetDevice(device_a));
    fill_kernel<<<(kStagingBytes + 255) / 256, 256, 0, stream_a>>>(
        static_cast<std::uint8_t*>(device_a_view), kStagingBytes, 0x3U);
    CHECK(cudaStreamSynchronize(stream_a));
    CHECK(cudaSetDevice(device_b));
    sum_kernel<<<(kStagingBytes + 255) / 256, 256, 0, stream_b>>>(
        static_cast<const std::uint8_t*>(peer_view), kStagingBytes, sum);
    CHECK(cudaStreamSynchronize(stream_b));

    const std::size_t sizes[] = {4ULL << 10, 256ULL << 10, 4ULL << 20, kStagingBytes};
    std::printf("%10s %14s %14s %12s\n", "bytes", "fill_ms", "peer_read_ms", "handoff_us");
    for (const std::size_t bytes : sizes) {
        const unsigned blocks = static_cast<unsigned>((bytes + 255) / 256);
        // Stage A: device A publishes into its own mapped pinned buffer.
        CHECK(cudaSetDevice(device_a));
        CHECK(cudaEventRecord(start, stream_a));
        fill_kernel<<<blocks, 256, 0, stream_a>>>(static_cast<std::uint8_t*>(device_a_view), bytes,
                                                 0x3U);
        CHECK(cudaEventRecord(stop, stream_a));
        CHECK(cudaEventSynchronize(stop));
        const float fill_ms = milliseconds(start, stop);

        // Handoff: A's work must complete before B reads the published bytes, which is the ordering
        // the arrival tokens enforce inside the kernel.
        const auto handoff_begin = std::chrono::steady_clock::now();

        // Stage B: device B reads the peer's mapped buffer through its own pointer into that buffer.
        CHECK(cudaSetDevice(device_b));
        CHECK(cudaMemsetAsync(sum, 0, sizeof(unsigned long long), stream_b));
        CHECK(cudaEventRecord(start, stream_b));
        sum_kernel<<<blocks, 256, 0, stream_b>>>(static_cast<const std::uint8_t*>(peer_view), bytes,
                                                 sum);
        CHECK(cudaEventRecord(stop, stream_b));
        CHECK(cudaEventSynchronize(stop));
        const float read_ms = milliseconds(start, stop);
        const auto handoff_end = std::chrono::steady_clock::now();
        const double handoff_us =
            std::chrono::duration<double, std::micro>(handoff_end - handoff_begin).count();

        unsigned long long observed = 0;
        CHECK(cudaMemcpy(&observed, sum, sizeof(observed), cudaMemcpyDeviceToHost));
        const unsigned long long expected =
            [](std::size_t n) {
                unsigned long long total = 0;
                for (std::size_t i = 0; i < n; ++i) { total += (0x3U + (i & 0xFFU)) & 0xFFU; }
                return total;
            }(bytes);
        const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        std::printf("%10zu %14.4f %14.4f %12.1f   %s (%.1f + %.1f GiB/s)  sum=%s\n", bytes, fill_ms,
                    read_ms, handoff_us, observed == expected ? "OK  " : "BAD ",
                    gib / (fill_ms / 1000.0), gib / (read_ms / 1000.0),
                    observed == expected ? "match" : "MISMATCH");
    }
    return 0;
}
