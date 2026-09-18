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

    const DeviceContext& a() const noexcept { return a_; }
    const DeviceContext& b() const noexcept { return b_; }
    bool p2p_available() const noexcept { return p2p_; }

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
    std::uint64_t ar_call_ = 0; // monotonically increasing arrival token
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
