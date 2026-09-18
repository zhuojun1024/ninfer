#include "core/tp/device_pair.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

// Find two distinct devices. Prefers two devices with the same name (the TP
// pair); falls back to any two devices for the communication contract.
std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    std::vector<std::string> names(count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        names[i] = prop.name;
    }
    for (int a = 0; a < count; ++a) {
        for (int b = a + 1; b < count; ++b) {
            if (names[a] == names[b]) { return {a, b}; }
        }
    }
    return {0, 1};
}

int check_allreduce(ninfer::tp::DevicePair& pair, std::size_t count_bytes) {
    const std::size_t elements = count_bytes / 2;
    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> ref_a(elements), ref_b(elements);
    std::vector<__nv_bfloat16> bf_a(elements), bf_b(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        ref_a[i] = dist(rng);
        ref_b[i] = dist(rng);
        bf_a[i]  = __float2bfloat16(ref_a[i]);
        bf_b[i]  = __float2bfloat16(ref_b[i]);
    }
    // DeviceBuffer allocates on the current device; each shard buffer must live
    // on its own device, so bind before each allocation and each read-back.
    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer buf_a(count_bytes);
    buf_a.copy_from_host(bf_a.data(), count_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer buf_b(count_bytes);
    buf_b.copy_from_host(bf_b.data(), count_bytes);

    // The host-staging allreduce enqueues its H2D copies on the caller's
    // compute streams; mirror production by driving it on per-device streams
    // and settling them before the default-stream read-back.
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);
    pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);

    pair.a().bind_to_current_thread();
    std::vector<__nv_bfloat16> got_a(elements);
    buf_a.copy_to_host(got_a.data(), count_bytes);
    pair.b().bind_to_current_thread();
    std::vector<__nv_bfloat16> got_b(elements);
    buf_b.copy_to_host(got_b.data(), count_bytes);

    int failures = 0;
    for (int which = 0; which < 2; ++which) {
        const auto& got = which == 0 ? got_a : got_b;
        for (std::size_t i = 0; i < elements; ++i) {
            const float expected = __bfloat162float(__float2bfloat16(ref_a[i])) +
                                   __bfloat162float(__float2bfloat16(ref_b[i]));
            const float actual   = __bfloat162float(got[i]);
            const float tol      = 0.01f * std::max(1.0f, std::abs(expected));
            if (std::abs(actual - expected) > tol) {
                if (failures < 4) {
                    std::cerr << "allreduce[" << which << "] element " << i << ": expected "
                              << expected << ", got " << actual << '\n';
                }
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver || count < 2) {
        std::cout << "SKIP: fewer than two CUDA devices\n";
        return 77;
    }
    const auto [dev_a, dev_b] = pick_devices();
    if (dev_a < 0) {
        std::cout << "SKIP: no usable device pair\n";
        return 77;
    }
    std::cout << "DevicePair(" << dev_a << ", " << dev_b << ")\n";

    int failures = 0;
    {
        ninfer::tp::DevicePair pair(dev_a, dev_b);
        std::cout << "p2p_available=" << pair.p2p_available() << '\n';
        // 20480 bytes = 10240 BF16 elements, the 27B hidden allreduce size.
        failures += check_allreduce(pair, 20480);
        // Small and larger shapes.
        failures += check_allreduce(pair, 16);
        failures += check_allreduce(pair, 1 << 20);
        // Move semantics: a moved-from pair must not retain p2p state.
        ninfer::tp::DevicePair moved(std::move(pair));
        if (pair.p2p_available()) {
            std::cerr << "moved-from pair retained p2p state\n";
            ++failures;
        }
        failures += check_allreduce(moved, 20480);
    }
    if (failures == 0) { std::cout << "PASS\n"; return 0; }
    std::cerr << failures << " failures\n";
    return 1;
}
