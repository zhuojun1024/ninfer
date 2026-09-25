#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

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

    // Spawns the in-kernel transport's diagnostics thread, which runs unless
    // NINFER_TP2_AR_WATCHDOG=0. It prints the mapped id/arrival/order state to stderr once per stall
    // episode, so a stuck in-kernel transport can be classified after the fact: ids that stopped
    // advancing with published arrivals mean a spin failed to observe the peer's write, and a stale
    // order slot means the slice write-order chain broke. Ids are host-authored (see
    // create_ar_channel), so the two sides can no longer diverge. The thread costs nothing
    // measurable: it sleeps 25 ms per tick, reads one host counter plus, while idle, the two mapped
    // stall flags, and touches the kernel-written arrival lines only after a device has tripped.
    void start_ar_watchdog();

    const DeviceContext& a() const noexcept { return a_; }
    const DeviceContext& b() const noexcept { return b_; }
    bool p2p_available() const noexcept { return p2p_; }
    // True when allreduce runs entirely in-kernel over mapped pinned host staging with no host
    // synchronization. Only this transport may be captured into a CUDA Graph: the P2P and
    // host-staging fallbacks synchronize the caller's streams.
    bool in_kernel_allreduce() const noexcept { return in_kernel_available_; }

    // Bounded-spin safety for the in-kernel transport. Every rendezvous spin carries a wall-clock
    // deadline (NINFER_TP2_AR_TIMEOUT_MS, default 10000, 0 disables it): a spin that can never be
    // satisfied would otherwise pin both devices at 100% forever and freeze the whole process. On
    // expiry the side that gave up raises its mapped flag and the kernel returns, so the caller's
    // streams drain and the request can fail instead of the service locking up. The caller must
    // synchronize both streams and call clear_ar_stall() before the next round, and must not keep
    // using state a stalled round may have half-written.
    [[nodiscard]] bool ar_stalled() const noexcept;
    // Clears both trip flags. The arrival and write-order slots need no repair: ids only ever
    // increase (see create_ar_channel), so a slot left over from the stalled round can never
    // satisfy a later spin. Both streams must be synchronized before this call.
    void clear_ar_stall() noexcept;
    // Id published by the most recent arming or eager collective, for diagnostics and error
    // messages. Equal on both sides by construction; it exists so a stall can name its round.
    [[nodiscard]] std::uint64_t ar_last_id() const noexcept;

    // Every collective carries an id that must be unique for the lifetime of the pair: a spin is
    // satisfied by any arrival slot that holds its id, so a reused id lets a kernel read the peer's
    // staging before the peer has written it. A captured graph takes its ids from a channel. The
    // channel's pinned cell is copied into the graph by a memcpy node, so every replay reads the id
    // block the host published for *that* launch, and the ordinal within the block is baked at
    // capture - it is a constant of the graph, which is why baking it is safe where a baked id
    // would not be. Eager calls draw from a separate monotonic range. One channel per captured
    // graph: two graphs launched in one round need disjoint id blocks, and two graphs sharing a
    // cell would read each other's base before their memcpy node ran.
    using ArChannel = std::uint32_t;
    static constexpr ArChannel kNoArChannel = 0;
    ArChannel create_ar_channel();
    // Brackets the collectives a capture records. They take the channel's next ordinals, and the
    // base memcpy node is recorded ahead of the first collective on each stream.
    void begin_capture(ArChannel channel);
    void end_capture();
    // Publishes the channel's next id block into its pinned cell and advances past it, so no id is
    // ever handed out twice. Call it immediately before launching the graph that owns the channel.
    void arm_round(ArChannel channel);

    // Fault injection for the bounded-spin test: skip the peer-side launch of the n-th in-kernel
    // collective issued after arming (1-based; 0 disarms). It desynchronizes the two sides exactly
    // like a divergent call sequence, which is the failure the bound exists for. Diagnostics only.
    void set_ar_fault_skip_peer_call(std::uint64_t serial) noexcept {
        ar_fault_skip_call_ = serial;
        ar_call_serial_     = 0;
    }

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
    // same staging, arrival ids and slicing - with a copy in place of the BF16 add, so it needs no
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
    // In-kernel allreduce (WSL2, no P2P): mapped pinned host staging + arrival ids. Each device's
    // kernel writes its delta to its own mapped host buffer, signals an arrival id, spins on the
    // peer's id, then reads the peer's buffer and sums in place - all inside the kernel, so the host
    // never calls cudaStreamSynchronize. Mapped pinned memory is verified available at construction;
    // when it is not, allreduce falls back to the host-staging path.
    bool in_kernel_available_ = false;
    std::size_t in_kernel_bytes_ = 0;
    void* host_a_ = nullptr; // cudaFreeHost handle for device a's staging
    void* host_b_ = nullptr; // cudaFreeHost handle for device b's staging
    void* dev_a_  = nullptr; // device-side pointer (device a) for host_a_
    void* dev_b_  = nullptr; // device-side pointer (device b) for host_b_
    // Arrival ids, one per allreduce slice, in mapped pinned host memory (the caller decides which
    // shard drives a pair, so a device allocation could be dereferenced from the other device's
    // context). Each slice writes its id, publishes an arrival, then spins for the peer's.
    unsigned long long* arrival_a_ = nullptr; // device pointer (device a) for its arrival array
    unsigned long long* arrival_b_ = nullptr; // device pointer (device b) for its arrival array
    void* arrival_host_a_ = nullptr; // cudaFreeHost handle for arrival_a_
    void* arrival_host_b_ = nullptr; // cudaFreeHost handle for arrival_b_
    // Bounded-spin state: one mapped flag per side, a cache line apart, plus the deadline handed to
    // every kernel launch (see ar_stalled).
    int* stall_host_ = nullptr; // cudaFreeHost handle: 2 * kArTokenStrideBytes
    int* stall_a_    = nullptr; // device pointer to side a's own flag
    int* stall_b_    = nullptr; // device pointer to side b's own flag
    unsigned long long ar_timeout_ns_ = 0; // 0 disables the deadline
    std::uint64_t ar_call_serial_     = 0; // in-kernel collectives issued since the last arming
    std::uint64_t ar_fault_skip_call_ = 0; // fault injection, see set_ar_fault_skip_peer_call
    // Rendezvous ids (see create_ar_channel). All channels share one pinned cell block and one
    // device scalar block per device; a channel's memcpy node copies its 8-byte cell into its own
    // scalar inside the captured graph.
    struct ArChannelState {
        unsigned long long* host = nullptr; // pinned cell: the graph's memcpy node reads it per replay
        void* dev_a = nullptr; // device scalar the a-side memcpy node writes
        void* dev_b = nullptr; // device scalar the b-side memcpy node writes
        std::uint64_t next_base = 0; // first id of the next arming; only ever increases
        std::uint32_t calls     = 0; // collectives numbered by the last capture
        bool seen_a             = false; // the base memcpy is recorded once per capture, per stream
        bool seen_b             = false;
    };
    static constexpr std::size_t kArMaxChannels = 16;
    unsigned long long* base_host_ = nullptr; // cudaFreeHost handle: kArMaxChannels id cells
    void* base_dev_a_ = nullptr; // cudaMalloc on device a: kArMaxChannels scalars
    void* base_dev_b_ = nullptr; // cudaMalloc on device b: kArMaxChannels scalars
    std::vector<ArChannelState> ar_channels_; // one per captured graph, created on demand
    bool capturing_              = false;
    ArChannel capture_channel_   = kNoArChannel;
    std::uint32_t capture_index_ = 0; // ordinal assigned to the next captured collective
    // Eager ids come from their own high range, far above every channel's, so the two can never
    // meet and a stale arrival slot can never match a later id from the other range.
    std::uint64_t ar_eager_next_ = 1ULL << 62;
    // Published id, read by the watchdog thread and by the engine's stall message.
    std::atomic<unsigned long long> ar_last_id_{0};
    // Decode-sized staging: a single token's [hidden, 1] delta is a few KiB, so the prefill-sized
    // slots are almost all waste, and sharing them would let a small call's parity slot alias a
    // prefill payload the peer is still reading.
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
    // True when the injected collective is the one whose peer launch must be skipped.
    bool fault_skip_peer();
    // What one collective runs with: the ordinal baked at capture (0 for eager, where the host
    // writes the id itself) and the base cell each side's kernel reads. Recording the base memcpy
    // node for a captured collective happens here, on its first appearance per stream.
    struct CollectiveId {
        // Captured: the base cell each side's memcpy node filled, plus the ordinal baked into this
        // kernel. Eager: no cell, and the id travels in the launch arguments, which are fresh on
        // every launch - a shared cell would be overwritten by the next call before the kernel ran.
        const unsigned long long* base_a = nullptr;
        const unsigned long long* base_b = nullptr;
        unsigned int call_index = 0;
        unsigned long long value = 0;
    };
    CollectiveId acquire_collective_id(cudaStream_t stream_a, cudaStream_t stream_b);
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
