#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::tp {

// Two independent DeviceContexts executing the same Op sequence in lockstep.
// The owning Program (Phase 3) drives both streams; this type owns the device
// facts and the inter-device communication path only.
class DevicePair {
public:
    explicit DevicePair(int device_a, int device_b);
    ~DevicePair();

    DevicePair(const DevicePair&)            = delete;
    DevicePair& operator=(const DevicePair&) = delete;
    DevicePair(DevicePair&& other) noexcept;
    DevicePair& operator=(DevicePair&& other) noexcept;

    // Spawns an env-gated (NINFER_TP2_AR_WATCHDOG=1) diagnostics thread that prints the mapped
    // token/arrival/order state to stderr twice a second, so a stuck in-kernel transport can be
    // classified after the fact: unequal tokens mean the two sides' call sequences diverged, equal
    // tokens with published arrivals mean a spin failed to observe the peer's write, and a stale
    // order slot means the slice write-order chain broke.
    void start_ar_watchdog();

    const DeviceContext& a() const noexcept { return a_; }
    const DeviceContext& b() const noexcept { return b_; }
    bool p2p_available() const noexcept { return p2p_; }
    // True when allreduce runs entirely in-kernel over mapped pinned host staging with no host
    // synchronization. Only this transport may be captured into a CUDA Graph: the P2P and
    // host-staging fallbacks synchronize the caller's streams.
    bool in_kernel_allreduce() const noexcept { return in_kernel_available_; }

    // In-place all-reduce of count_bytes (multiple of 16): on return both
    // buffers contain a + b elementwise. stream_a/stream_b are the compute
    // streams that produced data_a/data_b. The P2P path runs peer copies plus
    // an add kernel per device on the DevicePair's own streams and completes
    // them before returning. The host-staging path (WSL2 blocks peer access)
    // stages both halves through one pinned buffer on stream_a/stream_b: the
    // D2H copies are ordered after the producing kernels and the H2D copies
    // before the caller's next work on those streams, so only the two D2H
    // transfers are synchronized.
    void allreduce(void* data_a, void* data_b, std::size_t count_bytes,
                   cudaStream_t stream_a, cudaStream_t stream_b);

    // Byte-exact exchange of count_bytes (multiple of 16): on return recv_a holds send_b's bytes and
    // recv_b holds send_a's bytes, bit for bit. The in-kernel route reuses the all-reduce transport -
    // same staging, arrival tokens and slicing - with a copy in place of the BF16 add, so it needs no
    // host synchronization and is safe to capture into a CUDA Graph. Use it for any payload the BF16
    // add would reinterpret: integer ids, FP32 scores, or bits that must survive unchanged. A side may
    // pass the same buffer as its send and receive: it publishes its own bytes and returns the peer's,
    // because the two halves are distinct.
    void sendrecv(const void* send_a, void* recv_a, const void* send_b, void* recv_b,
                  std::size_t count_bytes, cudaStream_t stream_a, cudaStream_t stream_b);

private:
    DeviceContext a_;
    DeviceContext b_;
    bool p2p_ = false;
    std::unique_ptr<PinnedHostBuffer> staging_;
    // In-kernel allreduce (WSL2, no P2P): mapped pinned host staging + arrival
    // tokens. Each device's kernel writes its delta to its own mapped host
    // buffer, signals an arrival token, spins on the peer's token, then reads
    // the peer's buffer and sums in place - all inside the kernel, so the host
    // never calls cudaStreamSynchronize. Mapped pinned memory is verified
    // available at construction; when it is not, allreduce falls back to the
    // host-staging path.
    bool in_kernel_available_ = false;
    std::size_t in_kernel_bytes_ = 0;
    void* host_a_ = nullptr; // cudaFreeHost handle for device a's staging
    void* host_b_ = nullptr; // cudaFreeHost handle for device b's staging
    void* dev_a_  = nullptr; // device-side pointer (device a) for host_a_
    void* dev_b_  = nullptr; // device-side pointer (device b) for host_b_
    int*  arrival_a_ = nullptr; // device-side pointer (device a) for its arrival token
    int*  arrival_b_ = nullptr; // device-side pointer (device b) for its arrival token
    void* arrival_host_a_ = nullptr; // cudaFreeHost handle for arrival_a_
    void* arrival_host_b_ = nullptr; // cudaFreeHost handle for arrival_b_
    // Arrival tokens, one per allreduce caller side. Each allreduce call bumps its own side's token
    // with a one-thread kernel and the allreduce reads it back, so the token is replayable inside a
    // CUDA Graph: a value-baked kernel argument would be replayed unchanged and let a peer that is a
    // call ahead satisfy its spin from the previous replay. They live in mapped pinned host memory
    // rather than on a device heap because the caller decides which shard drives a pair: the stream
    // an allreduce runs on is not necessarily a_, and a device allocation would then be dereferenced
    // from the other device's context. The two counters live one cache line apart (see
    // kArTokenStrideBytes in device_pair.cu): both devices read-modify-write their own int on every
    // call, and adjacent ints on one line lose increments to the peer's non-coherent line
    // write-back, which desynchronizes the tokens and spins both devices forever.
    int* token_host_ = nullptr; // cudaFreeHost handle
    int* token_a_    = nullptr; // device pointer into token_host_
    int* token_b_    = nullptr; // device pointer into token_host_
    // Size-keyed transport policy for the in-kernel route. A decode-sized payload is a latency
    // problem: one block, one launch, and its own compact staging so the two parity slots never
    // alias the prefill-sized ones. A prefill-sized payload is a bandwidth problem and keeps the
    // sliced transport, which is already at the link floor.
    bool ar_size_keyed_   = true;
    bool small_available_ = false;
    void* small_host_a_ = nullptr; // cudaFreeHost handle for device a's decode-sized staging
    void* small_host_b_ = nullptr; // cudaFreeHost handle for device b's decode-sized staging
    void* small_dev_a_  = nullptr; // device-side pointer (device a) for small_host_a_
    void* small_dev_b_  = nullptr; // device-side pointer (device b) for small_host_b_
    // Diagnostics state for the in-kernel transport (see start_ar_watchdog). Held through a
    // shared_ptr because the detached watchdog thread only reads the mapped arrays and may outlive
    // moves of the pair.
    struct ArWatchState;
    std::shared_ptr<ArWatchState> watch_;
    void note_ar_call(std::size_t count_bytes);
    void stop_ar_watchdog();
};

// Non-owning view of one shard of a tensor split along dim. The local tensor
// is a valid contiguous view: for dim < 3 the shard keeps the full strides of
// the upper dims, so callers must treat it as a strided window, not a fresh
// allocation. For the 27B dense model only dims 0 and 1 are split (both
// produce contiguous shards of a contiguous parent).
struct ShardedTensor {
    Tensor local;
    int shard  = 0;
    int shards = 1;
};

// View of shard index of a contiguous tensor split along dim.
ShardedTensor shard_view(const Tensor& full, int dim, std::int32_t local_len, int shard);

// Column-parallel split of a [rows, cols] weight: shard gets cols/shards
// contiguous columns.
ShardedTensor column_shard(const Tensor& weight, int shard, int shards);

// Row-parallel split of a [rows, cols] weight: shard gets rows/shards
// contiguous rows.
ShardedTensor row_shard(const Tensor& weight, int shard, int shards);

} // namespace ninfer::tp
