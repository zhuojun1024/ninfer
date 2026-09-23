#include "core/tp/device_pair.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

// Byte-exact exchange: each shard must receive the peer's send buffer bit for bit, including lanes
// that spell a signaling NaN, which a BF16 add would quiet. The payloads mix adversarial 16-bit
// lanes with the I32 id / FP32 score bit patterns the split proposal head and the peer selector
// transport carry. Both the single-block and the sliced in-kernel paths are covered by the sizes.
int check_sendrecv(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int iterations) {
    const std::size_t words = count_bytes / 4;
    constexpr std::uint16_t kAdversarial[] = {0x7C01, 0x7C3F, 0xFC01, 0xFC3F,
                                              0xFFFF, 0x0001, 0x7F80, 0x0000};
    constexpr std::size_t kLanes = sizeof(kAdversarial) / sizeof(kAdversarial[0]);
    int failures = 0;
    std::vector<std::uint32_t> host_a(words), host_b(words), got(words);

    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer send_a(count_bytes), recv_a(count_bytes);
    cudaStream_t stream_a = nullptr;
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer send_b(count_bytes), recv_b(count_bytes);
    cudaStream_t stream_b = nullptr;
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    for (int iteration = 0; iteration < iterations && failures == 0; ++iteration) {
        for (std::size_t i = 0; i < words; ++i) {
            const std::uint16_t low = kAdversarial[(i + static_cast<std::size_t>(iteration)) % kLanes];
            const std::uint16_t high =
                static_cast<std::uint16_t>((i * 2654435761u + static_cast<std::size_t>(iteration) * 40503u) & 0xFFFFu);
            host_a[i] = static_cast<std::uint32_t>(low) | (static_cast<std::uint32_t>(high) << 16);
            host_b[i] = ~(host_a[i] + 0x9E3779B9u + static_cast<std::uint32_t>(iteration));
        }
        pair.a().bind_to_current_thread();
        cudaMemcpyAsync(send_a.p, host_a.data(), count_bytes, cudaMemcpyHostToDevice, stream_a);
        cudaMemsetAsync(recv_a.p, 0xA5, count_bytes, stream_a);
        pair.b().bind_to_current_thread();
        cudaMemcpyAsync(send_b.p, host_b.data(), count_bytes, cudaMemcpyHostToDevice, stream_b);
        cudaMemsetAsync(recv_b.p, 0x5A, count_bytes, stream_b);
        pair.sendrecv(send_a.p, recv_a.p, send_b.p, recv_b.p, count_bytes, stream_a, stream_b);
        pair.a().bind_to_current_thread();
        cudaMemcpyAsync(got.data(), recv_a.p, count_bytes, cudaMemcpyDeviceToHost, stream_a);
        cudaStreamSynchronize(stream_a);
        if (std::memcmp(got.data(), host_b.data(), count_bytes) != 0) {
            std::cerr << "sendrecv[" << count_bytes << "] iteration " << iteration
                      << ": shard a did not receive shard b's bytes\n";
            ++failures;
            break;
        }
        pair.b().bind_to_current_thread();
        cudaMemcpyAsync(got.data(), recv_b.p, count_bytes, cudaMemcpyDeviceToHost, stream_b);
        cudaStreamSynchronize(stream_b);
        if (std::memcmp(got.data(), host_a.data(), count_bytes) != 0) {
            std::cerr << "sendrecv[" << count_bytes << "] iteration " << iteration
                      << ": shard b did not receive shard a's bytes\n";
            ++failures;
            break;
        }
    }
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
    return failures;
}

// One captured send-receive queue replayed many times, with an eager all-reduce queued between
// replays (the production interleave of graphed verify windows and eager propose collectives).
// The graph body's receives are byte exact and identical every replay, so the final readback is
// checkable; a transport whose two sides ever disagree on the call sequence hangs the sync below,
// which is the regression this guards.
int check_graph_queue(ninfer::tp::DevicePair& pair, std::size_t graph_bytes, std::size_t eager_bytes,
                      int calls, int replays) {
    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer send_a(graph_bytes), recv_a(graph_bytes);
    ninfer::DeviceBuffer add_a(eager_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer send_b(graph_bytes), recv_b(graph_bytes);
    ninfer::DeviceBuffer add_b(eager_bytes);
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0xC0FFEEu + static_cast<unsigned>(graph_bytes % 65521u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    const std::size_t graph_elements = graph_bytes / 2;
    const std::size_t eager_elements = eager_bytes / 2;
    std::vector<__nv_bfloat16> host_a(graph_elements), host_b(graph_elements);
    std::vector<__nv_bfloat16> eager_a(eager_elements), eager_b(eager_elements);
    std::vector<__nv_bfloat16> got_a(graph_elements);
    for (std::size_t i = 0; i < graph_elements; ++i) {
        host_a[i] = __float2bfloat16(dist(rng));
        host_b[i] = __float2bfloat16(dist(rng));
    }
    for (std::size_t i = 0; i < eager_elements; ++i) {
        eager_a[i] = __float2bfloat16(dist(rng));
        eager_b[i] = __float2bfloat16(dist(rng));
    }
    pair.a().bind_to_current_thread();
    send_a.copy_from_host(host_a.data(), graph_bytes);
    add_a.copy_from_host(eager_a.data(), eager_bytes);
    pair.b().bind_to_current_thread();
    send_b.copy_from_host(host_b.data(), graph_bytes);
    add_b.copy_from_host(eager_b.data(), eager_bytes);

    // One rendezvous id channel for this graph. The host publishes a fresh id block before every
    // replay, which is what lets the transport tell two replays apart; a reused id would let the
    // peer's stale staging satisfy the spin.
    const auto channel = pair.create_ar_channel();
    cudaGraph_t graph_a = nullptr, graph_b = nullptr;
    cudaGraphExec_t exec_a = nullptr, exec_b = nullptr;
    pair.begin_capture(channel);
    pair.a().bind_to_current_thread();
    cudaStreamBeginCapture(stream_a, cudaStreamCaptureModeThreadLocal);
    pair.b().bind_to_current_thread();
    cudaStreamBeginCapture(stream_b, cudaStreamCaptureModeThreadLocal);
    pair.a().bind_to_current_thread();
    for (int call = 0; call < calls; ++call) {
        pair.sendrecv(send_a.p, recv_a.p, send_b.p, recv_b.p, graph_bytes, stream_a, stream_b);
    }
    cudaStreamEndCapture(stream_a, &graph_a);
    pair.b().bind_to_current_thread();
    cudaStreamEndCapture(stream_b, &graph_b);
    cudaGraphInstantiate(&exec_a, graph_a, 0);
    cudaGraphInstantiate(&exec_b, graph_b, 0);
    pair.end_capture();

    int failures = 0;
    for (int replay = 0; replay < replays; ++replay) {
        // Fresh operands every replay. This is what makes a stale rendezvous id observable: with the
        // same bytes every time, a replay that skipped the peer's write would still read a buffer
        // holding the expected values and the check would pass.
        for (std::size_t i = 0; i < graph_elements; ++i) {
            host_a[i] = __float2bfloat16(dist(rng));
            host_b[i] = __float2bfloat16(dist(rng));
        }
        pair.a().bind_to_current_thread();
        send_a.copy_from_host(host_a.data(), graph_bytes);
        pair.b().bind_to_current_thread();
        send_b.copy_from_host(host_b.data(), graph_bytes);
        // An eager collective interleaved with the replay: the two must draw ids from disjoint ranges.
        pair.allreduce(add_a.p, add_b.p, eager_bytes, stream_a, stream_b);
        pair.arm_round(channel);
        pair.a().bind_to_current_thread();
        cudaGraphLaunch(exec_a, stream_a);
        pair.b().bind_to_current_thread();
        cudaGraphLaunch(exec_b, stream_b);
        pair.a().bind_to_current_thread();
        cudaStreamSynchronize(stream_a);
        pair.b().bind_to_current_thread();
        cudaStreamSynchronize(stream_b);

        pair.a().bind_to_current_thread();
        recv_a.copy_to_host(got_a.data(), graph_bytes);
        for (std::size_t i = 0; i < graph_elements; ++i) {
            if (bits_of(got_a[i]) != bits_of(host_b[i])) {
                if (failures < 4) {
                    std::cerr << "graph queue[" << graph_bytes << "] replay " << replay << " element "
                              << i << ": expected " << __bfloat162float(host_b[i]) << ", got "
                              << __bfloat162float(got_a[i]) << '\n';
                }
                ++failures;
                break;
            }
        }
    }

    cudaGraphExecDestroy(exec_a);
    cudaGraphExecDestroy(exec_b);
    cudaGraphDestroy(graph_a);
    cudaGraphDestroy(graph_b);
    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return failures;
}

// Measures what the copy engines can do with the production large-payload shape, as the floor for
// the event-based large-payload path (PLAN Phase 3). That candidate moves exactly these bytes - a
// D2H of the local delta and an H2D of the peer's - so its cost cannot come in under this, and the
// sliced in-kernel path it would replace is already within 2% of the link's raw bound. Print-only,
// enabled by NINFER_TP2_AR_COPY_BENCH=1, because it measures a design decision rather than behavior.
int check_copy_engine_ceiling(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int iterations) {
    // Chunking and stream layout follow the candidate: the plan's chunk size, and one stream per
    // direction per device so both directions are in flight at once. That is what the candidate's
    // event handoff allows (an H2D starts as soon as the peer's chunk lands), and PCIe runs both
    // directions at once, so a probe that serializes them measures something the candidate need not do.
    const std::size_t chunk = std::min<std::size_t>(
        std::max<std::size_t>(count_bytes / 4, 512ULL << 10), 2ULL << 20);
    const int chunks = static_cast<int>((count_bytes + chunk - 1) / chunk);

    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer src_a(count_bytes);
    ninfer::DeviceBuffer scratch_a(count_bytes);
    void* host_a = nullptr;
    void* peer_a = nullptr;
    cudaHostAlloc(&host_a, count_bytes, cudaHostAllocPortable);
    cudaHostAlloc(&peer_a, count_bytes, cudaHostAllocPortable);
    cudaStream_t d2h_a = nullptr, h2d_a = nullptr;
    cudaStreamCreateWithFlags(&d2h_a, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&h2d_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer src_b(count_bytes);
    ninfer::DeviceBuffer scratch_b(count_bytes);
    void* host_b = nullptr;
    void* peer_b = nullptr;
    cudaHostAlloc(&host_b, count_bytes, cudaHostAllocPortable);
    cudaHostAlloc(&peer_b, count_bytes, cudaHostAllocPortable);
    cudaStream_t d2h_b = nullptr, h2d_b = nullptr;
    cudaStreamCreateWithFlags(&d2h_b, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&h2d_b, cudaStreamNonBlocking);

    const auto issue = [&] {
        pair.a().bind_to_current_thread();
        for (int c = 0; c < chunks; ++c) {
            const std::size_t offset = static_cast<std::size_t>(c) * chunk;
            const std::size_t bytes  = std::min(chunk, count_bytes - offset);
            cudaMemcpyAsync(static_cast<char*>(host_a) + offset,
                            static_cast<const char*>(src_a.p) + offset, bytes,
                            cudaMemcpyDeviceToHost, d2h_a);
            cudaMemcpyAsync(static_cast<char*>(scratch_a.p) + offset,
                            static_cast<const char*>(peer_a) + offset, bytes,
                            cudaMemcpyHostToDevice, h2d_a);
        }
        pair.b().bind_to_current_thread();
        for (int c = 0; c < chunks; ++c) {
            const std::size_t offset = static_cast<std::size_t>(c) * chunk;
            const std::size_t bytes  = std::min(chunk, count_bytes - offset);
            cudaMemcpyAsync(static_cast<char*>(host_b) + offset,
                            static_cast<const char*>(src_b.p) + offset, bytes,
                            cudaMemcpyDeviceToHost, d2h_b);
            cudaMemcpyAsync(static_cast<char*>(scratch_b.p) + offset,
                            static_cast<const char*>(peer_b) + offset, bytes,
                            cudaMemcpyHostToDevice, h2d_b);
        }
    };
    for (int i = 0; i < 5; ++i) { issue(); }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) { issue(); }
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(d2h_a);
    cudaStreamSynchronize(h2d_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(d2h_b);
    cudaStreamSynchronize(h2d_b);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() /
        iterations;
    const double gbps = 2.0 * static_cast<double>(count_bytes) / (ms * 1e6);
    std::cout << "copy engine ceiling[" << (count_bytes >> 20) << " MiB, " << chunks << " chunks]: "
              << ms << " ms per call (both directions), " << gbps
              << " GB/s counting both directions\n";

    pair.a().bind_to_current_thread();
    cudaFreeHost(host_a);
    cudaFreeHost(peer_a);
    cudaStreamDestroy(d2h_a);
    cudaStreamDestroy(h2d_a);
    pair.b().bind_to_current_thread();
    cudaFreeHost(host_b);
    cudaFreeHost(peer_b);
    cudaStreamDestroy(d2h_b);
    cudaStreamDestroy(h2d_b);
    return 0;
}

// A collective whose peer side never launches must give up on its deadline instead of spinning
// forever: the stall is reported, both streams still drain, and after clear_ar_stall() the same
// transport produces the elementwise sum again. The divergence is injected - the peer launch of one
// collective is skipped - which is the failure class the bound exists for.
int check_ar_timeout(ninfer::tp::DevicePair& pair, std::size_t count_bytes) {
    if (!pair.in_kernel_allreduce()) {
        std::cout << "SKIP ar timeout: the in-kernel transport is unavailable\n";
        return 0;
    }
    const std::size_t elements = count_bytes / 2;
    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer buf_a(count_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer buf_b(count_bytes);
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0x7E50u + static_cast<unsigned>(count_bytes % 65521u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<__nv_bfloat16> bf_a(elements), bf_b(elements);
    std::vector<__nv_bfloat16> got_a(elements), got_b(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        bf_a[i] = __float2bfloat16(dist(rng));
        bf_b[i] = __float2bfloat16(dist(rng));
    }
    pair.a().bind_to_current_thread();
    buf_a.copy_from_host(bf_a.data(), count_bytes);
    pair.b().bind_to_current_thread();
    buf_b.copy_from_host(bf_b.data(), count_bytes);

    int failures = 0;
    pair.set_ar_fault_skip_peer_call(1);
    const auto before = std::chrono::steady_clock::now();
    pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);
    const auto waited_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                             before)
            .count();
    const std::uint64_t stalled_id = pair.ar_last_id();
    std::cout << "ar timeout[" << count_bytes << "]: gave up after " << waited_ms
              << " ms at rendezvous id " << stalled_id << '\n';
    if (waited_ms < 100) {
        std::cerr << "ar timeout[" << count_bytes << "]: returned without waiting on the deadline\n";
        ++failures;
    }

    if (!pair.ar_stalled()) {
        std::cerr << "ar timeout[" << count_bytes << "]: the bounded spin did not report a stall\n";
        ++failures;
    }
    // While the pair is tripped, every further collective must bail at entry instead of waiting out
    // another deadline, or a desynchronized round would pay one timeout per allreduce.
    pair.set_ar_fault_skip_peer_call(0);
    const auto tripped_before = std::chrono::steady_clock::now();
    pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);
    const auto tripped_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                             tripped_before)
            .count();
    std::cout << "ar timeout[" << count_bytes << "]: tripped pair bailed after " << tripped_ms
              << " ms\n";
    if (tripped_ms > 500) {
        std::cerr << "ar timeout[" << count_bytes
                  << "]: a tripped pair waited again instead of bailing at entry\n";
        ++failures;
    }
    pair.clear_ar_stall();
    if (pair.ar_stalled()) {
        std::cerr << "ar timeout[" << count_bytes << "]: clear_ar_stall did not clear the trip\n";
        ++failures;
    }
    // The ids are what makes a stalled round recoverable in place: the next collective must not
    // reuse the id the stalled one ran with, or its spin could be satisfied by a stale arrival slot.
    if (pair.ar_last_id() == stalled_id) {
        std::cerr << "ar timeout[" << count_bytes << "]: the rendezvous id was reused after a stall\n";
        ++failures;
    }

    pair.set_ar_fault_skip_peer_call(0);
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
            std::cerr << "ar timeout[" << count_bytes << "]: element " << i
                      << " differs after re-arm\n";
            ++failures;
            break;
        }
    }

    if (failures == 0) {
        std::cout << "ar timeout[" << count_bytes << "]: stalled, re-armed, sum matches\n";
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
        // Byte-exact exchange at the shapes the split proposal head's candidate union and the peer
        // selector transport use, then at the small and sliced staging classes.
        failures += check_sendrecv(pair, 960, 64);
        failures += check_sendrecv(pair, 64, 64);
        failures += check_sendrecv(pair, 1 << 20, 16);
        // Captured queues replayed between eager calls: small fuse exchanges, sliced large
        // exchanges (write-order chain, separate token bump), and the two classes interleaved.
        failures += check_graph_queue(pair, 960, 20480, 32, 20);
        failures += check_graph_queue(pair, 2 << 20, 2 << 20, 8, 20);
        failures += check_graph_queue(pair, 960, 2 << 20, 16, 20);
        // A skipped peer launch must give up on the deadline and re-arm, not hang the process.
        failures += check_ar_timeout(pair, 20480);
        if (std::getenv("NINFER_TP2_AR_COPY_BENCH") != nullptr) {
            failures += check_copy_engine_ceiling(pair, 10 << 20, 40);
            failures += check_copy_engine_ceiling(pair, 24 << 20, 20);
        }
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
