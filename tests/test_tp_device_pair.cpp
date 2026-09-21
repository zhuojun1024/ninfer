#include "core/tp/device_pair.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
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

std::uint16_t bits_of(__nv_bfloat16 value) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// One payload size, repeated with fresh operands. The repetition is what makes the
// transport's ordering observable: a peer read that resolved to an earlier call's
// staging slot agrees with the current operands only by accident, while a read that
// resolved to the previous cache line of the same slot cannot agree at all. Every
// path (in-kernel, peer-copy and host staging) must produce the identical
// elementwise BF16 rounding add on both shards, so the check is bit for bit.
int check_allreduce(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int iterations) {
    const std::size_t elements = count_bytes / 2;
    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer buf_a(count_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer buf_b(count_bytes);

    // The allreduce enqueues on the caller's compute streams; mirror production by
    // driving it on per-device streams and settling them before the read-back.
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0x5EEDu + static_cast<unsigned>(count_bytes % 65521u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<__nv_bfloat16> bf_a(elements), bf_b(elements);
    std::vector<__nv_bfloat16> got_a(elements), got_b(elements);

    int failures = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t i = 0; i < elements; ++i) {
            bf_a[i] = __float2bfloat16(dist(rng));
            bf_b[i] = __float2bfloat16(dist(rng));
        }
        pair.a().bind_to_current_thread();
        buf_a.copy_from_host(bf_a.data(), count_bytes);
        pair.b().bind_to_current_thread();
        buf_b.copy_from_host(bf_b.data(), count_bytes);

        pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
        pair.a().bind_to_current_thread();
        cudaStreamSynchronize(stream_a);
        pair.b().bind_to_current_thread();
        cudaStreamSynchronize(stream_b);

        pair.a().bind_to_current_thread();
        buf_a.copy_to_host(got_a.data(), count_bytes);
        pair.b().bind_to_current_thread();
        buf_b.copy_to_host(got_b.data(), count_bytes);

        for (std::size_t i = 0; i < elements; ++i) {
            const __nv_bfloat16 expected =
                __float2bfloat16(__bfloat162float(bf_a[i]) + __bfloat162float(bf_b[i]));
            if (bits_of(got_a[i]) != bits_of(expected) || bits_of(got_b[i]) != bits_of(expected)) {
                if (failures < 4) {
                    std::cerr << "allreduce[" << count_bytes << "] iteration " << iteration
                              << " element " << i << ": expected " << __bfloat162float(expected)
                              << ", shard a " << __bfloat162float(got_a[i]) << ", shard b "
                              << __bfloat162float(got_b[i]) << '\n';
                }
                ++failures;
                break;
            }
        }
    }
    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return failures;
}

// The production pattern: many allreduces queued back to back on each shard's own
// stream, with no host synchronization between them. Each device then runs ahead of
// the other by as much as its own queue allows, which is the only condition under
// which the two parity slots of the mapped-host staging can be reused while the peer
// still reads them. Every call owns its own device buffers so the queued calls cannot
// overwrite each other's operands, and the whole queue is verified bit for bit.
int check_allreduce_queue(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int calls) {
    const std::size_t elements = count_bytes / 2;
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0xA11C0u + static_cast<unsigned>(count_bytes % 65521u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<__nv_bfloat16> bf_a(elements), bf_b(elements);
    std::vector<std::unique_ptr<ninfer::DeviceBuffer>> buf_a, buf_b;
    for (int call = 0; call < calls; ++call) {
        for (std::size_t i = 0; i < elements; ++i) {
            bf_a[i] = __float2bfloat16(dist(rng));
            bf_b[i] = __float2bfloat16(dist(rng));
        }
        pair.a().bind_to_current_thread();
        buf_a.push_back(std::make_unique<ninfer::DeviceBuffer>(count_bytes));
        buf_a.back()->copy_from_host(bf_a.data(), count_bytes);
        pair.b().bind_to_current_thread();
        buf_b.push_back(std::make_unique<ninfer::DeviceBuffer>(count_bytes));
        buf_b.back()->copy_from_host(bf_b.data(), count_bytes);
        pair.allreduce(buf_a.back()->p, buf_b.back()->p, count_bytes, stream_a, stream_b);
    }
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);

    // Re-derive the operands with the same generator so the check sees each call's
    // own inputs without holding them all on the host at once.
    std::mt19937 verify_rng(0xA11C0u + static_cast<unsigned>(count_bytes % 65521u));
    std::vector<__nv_bfloat16> expect_b(elements), got(elements);
    int failures = 0;
    for (int call = 0; call < calls; ++call) {
        for (std::size_t i = 0; i < elements; ++i) {
            bf_a[i] = __float2bfloat16(dist(verify_rng));
            bf_b[i] = __float2bfloat16(dist(verify_rng));
        }
        expect_b = bf_b;
        for (int which = 0; which < 2; ++which) {
            if (which == 0) {
                pair.a().bind_to_current_thread();
                buf_a[call]->copy_to_host(got.data(), count_bytes);
            } else {
                pair.b().bind_to_current_thread();
                buf_b[call]->copy_to_host(got.data(), count_bytes);
            }
            for (std::size_t i = 0; i < elements; ++i) {
                const __nv_bfloat16 expected =
                    __float2bfloat16(__bfloat162float(bf_a[i]) + __bfloat162float(expect_b[i]));
                if (bits_of(got[i]) != bits_of(expected)) {
                    if (failures < 4) {
                        std::cerr << "allreduce-queue[" << count_bytes << "] call " << call << " shard "
                                  << which << " element " << i << ": expected "
                                  << __bfloat162float(expected) << ", got "
                                  << __bfloat162float(got[i]) << '\n';
                    }
                    ++failures;
                    break;
                }
            }
        }
    }
    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
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
        failures += check_allreduce(pair, 20480, 64);
        // Small and larger shapes.
        failures += check_allreduce(pair, 16, 64);
        failures += check_allreduce(pair, 1 << 20, 64);
        // The decode window shapes the DFlash2 verify allreduces: one hidden
        // column per window position, then the full [vocab, width] logits merge
        // (152064 x 8 BF16 = 2433024 bytes, which the size-keyed transport splits
        // into five slices).
        failures += check_allreduce(pair, 81920, 64);
        failures += check_allreduce(pair, 2433024, 64);
        // The same shapes as a deep queue on both shards, which is how a layer
        // stack issues them.
        failures += check_allreduce_queue(pair, 20480, 200);
        failures += check_allreduce_queue(pair, 81920, 200);
        // Move semantics: a moved-from pair must not retain p2p state.
        ninfer::tp::DevicePair moved(std::move(pair));
        if (pair.p2p_available()) {
            std::cerr << "moved-from pair retained p2p state\n";
            ++failures;
        }
        failures += check_allreduce(moved, 20480, 8);
    }
    if (failures == 0) { std::cout << "PASS\n"; return 0; }
    std::cerr << failures << " failures\n";
    return 1;
}
