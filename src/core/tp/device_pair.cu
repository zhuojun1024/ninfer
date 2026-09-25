#include "core/tp/device_pair.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ninfer::tp {
namespace {

constexpr std::int32_t kAddThreads = 256;

// Elementwise in-place add of out += other for count_bytes of 2-byte BF16
// elements. Vectorized 8-byte (4-element) loads for aligned data.
__global__ void add_bf16_inplace(__nv_bfloat16* out, const __nv_bfloat16* other,
                                 std::size_t elements) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const std::size_t vec    = elements / 4;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vec; i += stride) {
        const uint2 o = reinterpret_cast<const uint2*>(other)[i];
        uint2 d       = reinterpret_cast<uint2*>(out)[i];
        const __nv_bfloat162 oh = *reinterpret_cast<const __nv_bfloat162*>(&o);
        const __nv_bfloat162 dh = *reinterpret_cast<const __nv_bfloat162*>(&d);
        const __nv_bfloat162 sum = __hadd2(dh, oh);
        reinterpret_cast<uint2*>(out)[i] = *reinterpret_cast<const uint2*>(&sum);
    }
    for (std::size_t i = vec * 4 + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += stride) {
        out[i] = __hadd(out[i], other[i]);
    }
}

std::size_t add_grid(std::size_t elements) {
    const std::size_t blocks = (elements + 4 * kAddThreads - 1) / (4 * kAddThreads);
    return std::max<std::size_t>(1, std::min<std::size_t>(blocks, 1024));
}

// Elementwise BF16 add of two 16-byte groups (eight BF16 values).
__device__ __forceinline__ uint4 add_bf16x8(uint4 a, uint4 b) {
    const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
    const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
    const __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(&a.z);
    const __nv_bfloat162 a3 = *reinterpret_cast<const __nv_bfloat162*>(&a.w);
    const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
    const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);
    const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&b.z);
    const __nv_bfloat162 b3 = *reinterpret_cast<const __nv_bfloat162*>(&b.w);
    uint4 r;
    *reinterpret_cast<__nv_bfloat162*>(&r.x) = __hadd2(a0, b0);
    *reinterpret_cast<__nv_bfloat162*>(&r.y) = __hadd2(a1, b1);
    *reinterpret_cast<__nv_bfloat162*>(&r.z) = __hadd2(a2, b2);
    *reinterpret_cast<__nv_bfloat162*>(&r.w) = __hadd2(a3, b3);
    return r;
}

// In-kernel all-reduce (WSL2, no P2P): both devices run this kernel on their
// own compute streams. Phase 1 writes the local delta into this device's
// mapped pinned host buffer; phase 2 has one thread per block publish an arrival
// token and spin (with __nanosleep) until the peer's token matches; phase 3
// reads the peer's buffer and writes the elementwise sum back in place.
// Cross-GPU synchronization happens entirely inside the kernel via the
// host-memory arrival tokens, so the host never calls cudaStreamSynchronize.
// The token is a monotonically increasing call number (never reset), so a stale
// token from a previous call can never satisfy the spin.
//
// A batched prefill all-reduces a [hidden, T] delta (megabytes), so the payload is moved in
// 16-byte groups when every pointer is group-aligned ('groups' is then count/8, zero otherwise and
// the scalar loops below cover the whole payload). Phase 3 reads four groups per thread per step
// because a dependent single-group loop over a mapped host buffer is latency-bound long before it
// is bandwidth-bound: the four loads issue before any of them is consumed.
//
// Slices: a decode-sized payload rides a single block, because its cost is the peer round trip, not
// the transfer. A prefill-sized payload is split over several blocks, each owning one contiguous
// slice and one arrival slot, so the slices reduce independently once their data has landed. On
// this host the payload's device link (device 1 sits behind a Gen4 x4 chipset link, ~7.9 GB/s in
// both directions combined) is the limit, and moving a slice out of the write queue early is what
// lets the peer read it while the remaining slices are still going out: see ar_slices. Block b
// additionally waits for its predecessor's slice to become system-visible before it writes, so the
// slices land in index order instead of interleaving into one long drain.
// block.b write order chain: block b only starts writing once block b-1 published its arrival, so
// each slice becomes system-visible in index order and the peer can read early slices while later
// ones are still in flight. Without it the concurrent writes interleave and no slice is complete
// until the whole transfer drains.
// The arrival token is read from device memory rather than passed by value: a CUDA Graph records
// kernel arguments, so a value-baked token would be replayed unchanged and a peer that is a call
// ahead would satisfy its spin from the previous replay. A device counter that every replay
// increments on both devices keeps the monotonic-token invariant inside a captured sequence.
// kAdd selects the reduction: true is the in-place all-reduce the TP-2 routes run, false is a
// byte-exact exchange where `out` receives the peer's staged bytes unchanged. Both share the
// staging, arrival-token and slice machinery; only phase 3 differs. The copy mode preserves every
// 16-bit lane, so a payload whose lanes spell a signaling NaN - an I32 id, an FP32 score - travels
// intact where the BF16 add would quiet it.
// Rendezvous ids are host-authored (see DevicePair::create_ar_channel): the base cell is written
// before the launch, a captured graph reading it through its own memcpy node and an eager call
// through the host's mapped cell. The ordinal within the block is baked at capture, where it is a
// constant of the graph. Ids therefore only ever increase and are never reused, which is what lets
// the arrival slots be compared for equality at all: a reused id would let a stale slot satisfy the
// spin before the peer wrote anything.
//
// Wall-clock nanoseconds for the bounded spin: %globaltimer is a device-wide clock every SM agrees
// on, unlike clock64(), whose per-SM counters drift apart under boost.
__device__ __forceinline__ unsigned long long ar_now_ns() {
    unsigned long long t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// True once the deadline taken at kernel entry has passed. A zero deadline disables the bound, so a
// timeout of 0 keeps the transport's behavior exactly as it was before the bound existed.
__device__ __forceinline__ bool ar_expired(unsigned long long deadline) {
    return deadline != 0ULL && ar_now_ns() >= deadline;
}

template <bool kAdd>
__global__ void ar_exchange(const __nv_bfloat16* local, __nv_bfloat16* out,
                            char* host_mine_base, const char* host_other_base, int count,
                            unsigned long long* arrival_mine, unsigned long long* arrival_other,
                            unsigned long long* order_mine, const unsigned long long* base_dev,
                            unsigned int call_index, unsigned long long base_value, int slot_bytes,
                            int groups, int* stall_mine, int* stall_peer,
                            unsigned long long timeout_ns) {
    // The id is a device load of a cell the host filled before the launch, plus the ordinal this
    // kernel was captured with. It is fixed for the whole call, so one read per block, published
    // through shared memory, is enough.
    __shared__ unsigned long long shared_token;
    // Cooperative give-up: thread 0 is the only spinner, so a timeout is published through shared
    // memory and every thread leaves the block together. Returning from one thread alone would
    // deadlock the block at its next __syncthreads.
    __shared__ int stalled;
    const unsigned long long deadline = timeout_ns == 0ULL ? 0ULL : ar_now_ns() + timeout_ns;
    // A tripped pair is one call out of step, so every later rendezvous would wait out its whole
    // deadline: without this entry check a desynchronized round pays one timeout per allreduce
    // (about 128 of them) before the host can react. Both flags are read, because either side may
    // have been the one that gave up.
    if (threadIdx.x == 0) {
        stalled = (*stall_mine != 0 || *stall_peer != 0) ? 1 : 0;
        if (stalled == 0) {
            // A captured collective reads the cell its graph's memcpy node filled and adds the
            // ordinal baked into this kernel; an eager launch carries its id in the launch
            // arguments, which are fresh on every launch.
            shared_token = base_dev != nullptr ? *base_dev + call_index : base_value;
        }
    }
    __syncthreads();
    if (stalled != 0) { return; }
    const unsigned long long token = shared_token;
    const int stride = blockDim.x;
    const int blocks = gridDim.x;
    const int tail   = groups * 8;
    auto* host_mine        = reinterpret_cast<__nv_bfloat16*>(
        host_mine_base + static_cast<std::size_t>(token & 1) * static_cast<std::size_t>(slot_bytes));
    const auto* host_other = reinterpret_cast<const __nv_bfloat16*>(
        host_other_base + static_cast<std::size_t>(token & 1) * static_cast<std::size_t>(slot_bytes));
    // This block's contiguous slice of the 16-byte groups. Slice bounds are derived from the block
    // index alone, so every block on this device and its peer agree on which slot covers which data.
    const int begin = static_cast<int>((static_cast<long long>(groups) * blockIdx.x) / blocks);
    const int end =
        static_cast<int>((static_cast<long long>(groups) * (blockIdx.x + 1)) / blocks);
    auto* mine_groups         = reinterpret_cast<uint4*>(host_mine);
    const auto* local_groups  = reinterpret_cast<const uint4*>(local);
    const auto* other_groups  = reinterpret_cast<const uint4*>(host_other);
    auto* out_groups          = reinterpret_cast<uint4*>(out);
    if (blocks > 1 && blockIdx.x > 0 && threadIdx.x == 0) {
        while (*(const volatile unsigned long long*)(order_mine + blockIdx.x - 1) != token) {
            if (ar_expired(deadline)) {
                stalled    = 1;
                *(volatile int*)stall_mine = 1;
                break;
            }
            __nanosleep(100);
        }
    }
    if (blocks > 1) {
        __syncthreads();
        if (stalled != 0) { return; }
    }
    for (int i = begin + threadIdx.x; i < end; i += stride) { mine_groups[i] = local_groups[i]; }
    if (blockIdx.x == 0) {
        for (int i = tail + threadIdx.x; i < count; i += stride) { host_mine[i] = local[i]; }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile unsigned long long*)(order_mine + blockIdx.x)  = token;
        *(volatile unsigned long long*)(arrival_mine + blockIdx.x) = token;
        __threadfence_system();
        while (*(const volatile unsigned long long*)(arrival_other + blockIdx.x) != token) {
            if (ar_expired(deadline)) {
                stalled    = 1;
                *(volatile int*)stall_mine = 1;
                break;
            }
            __nanosleep(100);
        }
    }
    __syncthreads();
    if (stalled != 0) { return; }
    __threadfence_system();
    int i = begin + threadIdx.x;
    for (; i + 3 * stride < end; i += 4 * stride) {
        const uint4 o0 = other_groups[i];
        const uint4 o1 = other_groups[i + stride];
        const uint4 o2 = other_groups[i + 2 * stride];
        const uint4 o3 = other_groups[i + 3 * stride];
        out_groups[i]              = kAdd ? add_bf16x8(local_groups[i], o0) : o0;
        out_groups[i + stride]     = kAdd ? add_bf16x8(local_groups[i + stride], o1) : o1;
        out_groups[i + 2 * stride] = kAdd ? add_bf16x8(local_groups[i + 2 * stride], o2) : o2;
        out_groups[i + 3 * stride] = kAdd ? add_bf16x8(local_groups[i + 3 * stride], o3) : o3;
    }
    for (; i < end; i += stride) {
        out_groups[i] = kAdd ? add_bf16x8(local_groups[i], other_groups[i]) : other_groups[i];
    }
    if (blockIdx.x == 0) {
        for (int j = tail + threadIdx.x; j < count; j += stride) {
            out[j] = kAdd ? __hadd(local[j], host_other[j]) : host_other[j];
        }
    }
}

// Staging capacity per device. A batched prefill reduces a [hidden, T] BF16 delta per layer
// (hidden=5120, T up to the maximum prefill chunk), so 24 MiB carries T up to 2048; exceeding the
// staging bound would silently fall back to the host-staging path, which synchronizes both compute
// streams on every layer.
constexpr std::size_t kInKernelArBytes = 24ULL << 20; // 24 MiB staging per device, per buffer
constexpr int kArThreads               = 1024;
constexpr int kArMaxSlices             = 8;
// A slot holds one rendezvous id as an unsigned long long: the ids are host-authored and never
// repeat over the pair's lifetime, which is only expressible in 64 bits - a 32-bit slot would let a
// stale value from one id range match a future id from another.
constexpr std::size_t kArSlotBytes     = 128; // one cache line per slot array (kArMaxSlices ids)
constexpr std::size_t kArTokenBytes    = 2 * kArSlotBytes; // arrival array, then write-order chain
constexpr std::size_t kArSlotIds       = kArSlotBytes / sizeof(unsigned long long);
// One cache line per side's bounded-spin flag: the mapped host path is not coherent between peers,
// so two flags sharing one line would let a peer's line write-back drop the other's store.
constexpr std::size_t kArTokenStrideBytes = 64;

// Decode-sized staging for the size-keyed transport. A single token's [hidden, 1] delta is a few
// KiB: the 24 MiB prefill slots are almost all waste, and sharing them would let a small call's
// parity slot alias a prefill payload the peer is still reading. This buffer holds one small
// payload per parity instead.
constexpr std::size_t kArSmallBytes = 64ULL << 10; // staging per device, per parity

// Slices per allreduce. A decode-sized payload is a latency problem and rides one slice. A
// prefill-sized payload is split into 512 KiB slices whose writes are ordered by the kernel's own
// chain, which lets the peer read early slices while the rest is still in flight; measured on the
// RTX 5060 Ti pair (T=1024: 10 MiB delta, device 1 behind a Gen4 x4 chipset link) this took the
// allreduce from 3.18 ms to 2.70 ms, essentially the link's two-way bandwidth bound (21 MiB at
// 7.9 GB/s = 2.66 ms). More slices lose to the per-slice handshake latency (16 slices: 3.84 ms,
// 32 slices: 6.24 ms), so the count stays small.
int ar_slices(std::size_t count_bytes) {
    if (count_bytes <= (256ULL << 10)) { return 1; }
    const std::size_t want = (count_bytes + (512ULL << 10) - 1) / (512ULL << 10);
    return static_cast<int>(std::min<std::size_t>(std::max<std::size_t>(want, 2),
                                                  static_cast<std::size_t>(kArMaxSlices)));
}

void require_bytes(std::size_t count_bytes) {
    if (count_bytes == 0) { return; }
    if (count_bytes % 16 != 0) {
        throw std::invalid_argument("tp allreduce: byte count must be a multiple of 16");
    }
}

} // namespace

// Watched state for the spin diagnostics: the host-visible token counters and the two arrival/order
// arrays of the in-kernel transport. The watchdog thread reads them with memcpy because the devices
// write them concurrently.
struct DevicePair::ArWatchState {
    const std::atomic<unsigned long long>* id = nullptr;
    const unsigned char* arrival_a = nullptr;
    const unsigned char* arrival_b = nullptr;
    std::atomic<unsigned long long> calls      = 0;
    std::atomic<unsigned long long> last_bytes = 0;
    std::atomic<bool> stop = false;
    std::thread thread;
};

// Stops and joins this pair's watchdog before the mapped arrays are freed. The thread holds only a
// raw pointer to the state block, which outlives it because the join happens first.
void DevicePair::stop_ar_watchdog() {
    if (!watch_) { return; }
    watch_->stop.store(true);
    if (watch_->thread.joinable()) { watch_->thread.join(); }
}

namespace {

// One arrival/order slot: an id per slice at the front of the array, the write-order chain for the
// same slices at byte kArSlotBytes (see the ar_exchange launch sites). Read with memcpy because the
// devices write them while the watchdog thread reads.
unsigned long long watch_slot(const unsigned char* base, int index) {
    unsigned long long value = 0;
    std::memcpy(&value, base + static_cast<std::size_t>(index) * sizeof(value), sizeof(value));
    return value;
}

// Renders eight slots for the watchdog's single-line dump.
void format_slots(const unsigned long long* values, char* out, std::size_t size) {
    int written = 0;
    for (int i = 0; i < 8; ++i) {
        written += std::snprintf(out + written, size - static_cast<std::size_t>(written), "%s%llu",
                                 i == 0 ? "" : " ", values[i]);
    }
}

} // namespace

void DevicePair::note_ar_call(std::size_t count_bytes) {
    if (!watch_) { return; }
    watch_->last_bytes.store(count_bytes, std::memory_order_relaxed);
    watch_->calls.fetch_add(1, std::memory_order_relaxed);
}

// Fault injection: only the armed process counts collectives, so an unarmed process pays nothing.
bool DevicePair::fault_skip_peer() {
    if (ar_fault_skip_call_ == 0) { return false; }
    ++ar_call_serial_;
    return ar_call_serial_ == ar_fault_skip_call_;
}

bool DevicePair::ar_stalled() const noexcept {
    if (stall_host_ == nullptr) { return false; }
    const auto* flags = reinterpret_cast<const int*>(stall_host_);
    return flags[0] != 0 || flags[kArTokenStrideBytes / sizeof(int)] != 0;
}

std::uint64_t DevicePair::ar_last_id() const noexcept {
    return static_cast<std::uint64_t>(ar_last_id_.load(std::memory_order_relaxed));
}

void DevicePair::clear_ar_stall() noexcept {
    if (stall_host_ == nullptr) { return; }
    std::memset(stall_host_, 0, 2 * kArTokenStrideBytes);
    // The arrival and write-order slots need no repair, and deliberately get none: no id is ever
    // handed out twice, so a slot left over from the stalled round can never satisfy a later spin.
    // That is what makes a stalled round recoverable in place instead of needing a renumbering.
}

DevicePair::ArChannel DevicePair::create_ar_channel() {
    if (base_host_ == nullptr || base_dev_a_ == nullptr || base_dev_b_ == nullptr) {
        throw std::logic_error("tp allreduce: the in-kernel transport is unavailable");
    }
    if (ar_channels_.size() >= kArMaxChannels) {
        throw std::logic_error("tp allreduce: the captured graphs exceed the rendezvous id channels");
    }
    const std::size_t index = ar_channels_.size();
    ArChannelState state;
    state.host  = base_host_ + index;
    state.dev_a = static_cast<char*>(base_dev_a_) + index * sizeof(unsigned long long);
    state.dev_b = static_cast<char*>(base_dev_b_) + index * sizeof(unsigned long long);
    // Channel k owns ids from (k + 1) << 32 upward, so the channels' blocks cannot overlap and
    // every id a channel hands out is larger than the ones it handed out before.
    state.next_base = (static_cast<std::uint64_t>(index) + 1) << 32;
    ar_channels_.push_back(state);
    return static_cast<ArChannel>(ar_channels_.size());
}

void DevicePair::begin_capture(ArChannel channel) {
    if (capturing_) { throw std::logic_error("tp allreduce: a capture is already open"); }
    if (channel == kNoArChannel || channel > ar_channels_.size()) {
        throw std::invalid_argument("tp allreduce: unknown rendezvous id channel");
    }
    capturing_            = true;
    capture_channel_      = channel;
    capture_index_        = 0;
    ArChannelState& state = ar_channels_[channel - 1];
    state.seen_a          = false;
    state.seen_b          = false;
}

void DevicePair::end_capture() {
    if (!capturing_) { return; }
    ar_channels_[capture_channel_ - 1].calls = capture_index_;
    capturing_                               = false;
    capture_channel_                         = kNoArChannel;
}

void DevicePair::arm_round(ArChannel channel) {
    if (channel == kNoArChannel || channel > ar_channels_.size()) {
        throw std::invalid_argument("tp allreduce: unknown rendezvous id channel");
    }
    ArChannelState& state = ar_channels_[channel - 1];
    *state.host           = static_cast<unsigned long long>(state.next_base);
    ar_last_id_.store(state.next_base, std::memory_order_relaxed);
    // Advance past this launch's block even when the capture numbered no collective: handing the
    // same block to a later launch would let a stale arrival slot satisfy its spin.
    state.next_base += state.calls == 0 ? 1 : state.calls;
}

DevicePair::CollectiveId DevicePair::acquire_collective_id(cudaStream_t stream_a,
                                                           cudaStream_t stream_b) {
    CollectiveId id;
    if (!capturing_) {
        // Eager: the id travels in the launch arguments, so there is nothing the next call could
        // overwrite before this kernel runs.
        id.value = ar_eager_next_++;
        ar_last_id_.store(id.value, std::memory_order_relaxed);
        return id;
    }
    ArChannelState& state = ar_channels_[capture_channel_ - 1];
    // The base has to arrive through the graph rather than as a kernel argument: a baked id would be
    // replayed unchanged, and a stale arrival slot holding it would satisfy the spin without the
    // peer having written anything.
    if (!state.seen_a) {
        a_.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(state.dev_a, state.host, sizeof(unsigned long long),
                                   cudaMemcpyHostToDevice, stream_a));
        state.seen_a = true;
    }
    if (!state.seen_b) {
        b_.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(state.dev_b, state.host, sizeof(unsigned long long),
                                   cudaMemcpyHostToDevice, stream_b));
        state.seen_b = true;
    }
    id.base_a     = static_cast<const unsigned long long*>(state.dev_a);
    id.base_b     = static_cast<const unsigned long long*>(state.dev_b);
    id.call_index = capture_index_++;
    return id;
}

void DevicePair::start_ar_watchdog() {
    if (!watch_ || std::getenv("NINFER_TP2_AR_WATCHDOG") == nullptr) { return; }
    ArWatchState* raw = watch_.get();
    watch_->thread    = std::thread([raw] {
        // The mapped arrays are read only while the transport is provably stalled (the host-side
        // call counter not advancing for a tick), so a healthy run's token/arrival lines see no
        // extra traffic from the diagnostics themselves.
        unsigned long long last_calls = 0;
        bool stalled                  = false;
        while (!raw->stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (raw->stop.load(std::memory_order_relaxed)) { break; }
            const unsigned long long calls = raw->calls.load(std::memory_order_relaxed);
            if (calls == last_calls) {
                // Before the first collective there is nothing to diagnose, and the model load makes
                // that the common case: skipping it keeps a dump meaningful instead of ~100 lines of
                // empty slots between the load and the first request.
                if (calls == 0) { continue; }
                if (!stalled) {
                    stalled = true;
                    std::fprintf(stderr, "[ar-watch] stalled at calls=%llu\n", calls);
                }
            } else {
                last_calls = calls;
                stalled    = false;
                continue;
            }
            const unsigned long long id = raw->id->load(std::memory_order_relaxed);
            unsigned long long slots[4][8] = {};
            for (int i = 0; i < 8; ++i) {
                slots[0][i] = watch_slot(raw->arrival_a, i);
                slots[1][i] = watch_slot(raw->arrival_a, static_cast<int>(kArSlotIds) + i);
                slots[2][i] = watch_slot(raw->arrival_b, i);
                slots[3][i] = watch_slot(raw->arrival_b, static_cast<int>(kArSlotIds) + i);
            }
            char arr_a[160] = {}, ord_a[160] = {}, arr_b[160] = {}, ord_b[160] = {};
            format_slots(slots[0], arr_a, sizeof(arr_a));
            format_slots(slots[1], ord_a, sizeof(ord_a));
            format_slots(slots[2], arr_b, sizeof(arr_b));
            format_slots(slots[3], ord_b, sizeof(ord_b));
            std::fprintf(stderr,
                         "[ar-watch] calls=%llu last=%llu id=%llu arrA=[%s] ordA=[%s] "
                         "arrB=[%s] ordB=[%s]\n",
                         raw->calls.load(), raw->last_bytes.load(), id, arr_a, ord_a, arr_b, ord_b);
        }
    });}

DevicePair::DevicePair(int device_a, int device_b) : a_(device_a), b_(device_b) {
    if (device_a == device_b) {
        throw std::invalid_argument("tp DevicePair: devices must be distinct");
    }
    ar_timeout_ns_ = 2000ULL * 1000000ULL; // 2 s: ~700x the slowest measured collective (2.7 ms)
    if (const char* timeout = std::getenv("NINFER_TP2_AR_TIMEOUT_MS")) {
        ar_timeout_ns_ = static_cast<unsigned long long>(std::strtoull(timeout, nullptr, 10)) *
                         1000000ULL;
    }
    // Fault injection for the bounded-spin acceptance run: skip the peer launch of one collective so
    // the pair desynchronizes exactly as a divergent call sequence would. Diagnostics only.
    if (const char* fault = std::getenv("NINFER_TP2_AR_FAULT_SKIP_PEER_CALL")) {
        ar_fault_skip_call_ = std::strtoull(fault, nullptr, 10);
    }
    int can_a_to_b = 0;
    int can_b_to_a = 0;
    const cudaError_t err_a = cudaDeviceCanAccessPeer(&can_a_to_b, device_a, device_b);
    const cudaError_t err_b = cudaDeviceCanAccessPeer(&can_b_to_a, device_b, device_a);
    if (err_a == cudaSuccess && err_b == cudaSuccess && can_a_to_b && can_b_to_a) {
        a_.bind_to_current_thread();
        b_.bind_to_current_thread();
        CUDA_CHECK(cudaDeviceEnablePeerAccess(device_b, 0));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(device_a, 0));
        p2p_ = true;
    }
    if (!p2p_) {
        // Set up the in-kernel allreduce path: mapped pinned staging buffers
        // (one per device) plus arrival tokens. cudaHostAllocMapped makes the
        // host memory directly addressable by the device (verified available on
        // WSL2); if it is not, allreduce falls back to the host-staging path.
        // The staging area is double buffered by allreduce call parity: the peer publishes its
        // arrival token before it reads its slice, so without two buffers a device could overwrite
        // the buffer the peer is still reading from the previous layer.
        a_.bind_to_current_thread();
        if (cudaHostAlloc(&host_a_, 2 * kInKernelArBytes,
                          cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
            cudaHostGetDevicePointer(&dev_a_, host_a_, 0) == cudaSuccess) {
            b_.bind_to_current_thread();
            if (cudaHostAlloc(&host_b_, 2 * kInKernelArBytes,
                              cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer(&dev_b_, host_b_, 0) == cudaSuccess) {
                a_.bind_to_current_thread();
                if (cudaHostAlloc(&arrival_host_a_, kArTokenBytes, cudaHostAllocPortable |
                                     cudaHostAllocMapped) == cudaSuccess &&
                    cudaHostGetDevicePointer((void**)&arrival_a_, arrival_host_a_, 0) == cudaSuccess) {
                    b_.bind_to_current_thread();
                    if (cudaHostAlloc(&arrival_host_b_, kArTokenBytes, cudaHostAllocPortable |
                                         cudaHostAllocMapped) == cudaSuccess &&
                        cudaHostGetDevicePointer((void**)&arrival_b_, arrival_host_b_, 0) == cudaSuccess) {
                        // The rendezvous id cells and the bounded-spin flags. A channel cell is
                        // host memory whose device scalars are written by the graph's memcpy node;
                        // the eager cell is mapped so the eager kernels read it directly.
                        bool ids = false;
                        if (cudaHostAlloc(reinterpret_cast<void**>(&stall_host_),
                                          2 * kArTokenStrideBytes, cudaHostAllocPortable |
                                              cudaHostAllocMapped) == cudaSuccess) {
                            std::memset(stall_host_, 0, 2 * kArTokenStrideBytes);
                            ids = cudaHostGetDevicePointer(reinterpret_cast<void**>(&stall_a_),
                                                           stall_host_, 0) == cudaSuccess;
                            if (ids) {
                                stall_b_ = reinterpret_cast<int*>(
                                    reinterpret_cast<char*>(static_cast<void*>(stall_a_)) +
                                    kArTokenStrideBytes);
                            } else {
                                stall_a_ = nullptr;
                            }
                        } else {
                            stall_host_ = nullptr;
                        }
                        if (ids) {
                            ids = cudaHostAlloc(reinterpret_cast<void**>(&base_host_),
                                                kArMaxChannels * sizeof(unsigned long long),
                                                cudaHostAllocPortable) == cudaSuccess;
                            if (ids) {
                                std::memset(base_host_, 0,
                                            kArMaxChannels * sizeof(unsigned long long));
                                a_.bind_to_current_thread();
                                ids = cudaMalloc(&base_dev_a_, kArMaxChannels *
                                                                   sizeof(unsigned long long)) ==
                                      cudaSuccess;
                                b_.bind_to_current_thread();
                                ids = ids && cudaMalloc(&base_dev_b_, kArMaxChannels *
                                                                          sizeof(unsigned long long)) ==
                                                 cudaSuccess;
                            }
                        }
                        if (ids) {
                            std::memset(arrival_host_a_, 0, kArTokenBytes);
                            std::memset(arrival_host_b_, 0, kArTokenBytes);
                            in_kernel_bytes_ = kInKernelArBytes;
                            in_kernel_available_ = true;
                        }
                    }
                }
            }
        }
        // Decode-sized staging, sized so its two parity slots never alias the prefill-sized ones.
        if (in_kernel_available_) {
            a_.bind_to_current_thread();
            if (cudaHostAlloc(&small_host_a_, 2 * kArSmallBytes,
                              cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer(&small_dev_a_, small_host_a_, 0) == cudaSuccess) {
                b_.bind_to_current_thread();
                if (cudaHostAlloc(&small_host_b_, 2 * kArSmallBytes,
                                  cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                    cudaHostGetDevicePointer(&small_dev_b_, small_host_b_, 0) == cudaSuccess) {
                    small_available_ = true;
                }
            }
        }
        if (!small_available_) {
            if (small_host_a_) { cudaFreeHost(small_host_a_); small_host_a_ = nullptr; small_dev_a_ = nullptr; }
            if (small_host_b_) { cudaFreeHost(small_host_b_); small_host_b_ = nullptr; small_dev_b_ = nullptr; }
        }
        // On any failure, free whatever was allocated and stay on host staging.
        if (!in_kernel_available_) {
            if (host_a_) { cudaFreeHost(host_a_); host_a_ = nullptr; dev_a_ = nullptr; }
            if (host_b_) { cudaFreeHost(host_b_); host_b_ = nullptr; dev_b_ = nullptr; }
            if (arrival_host_a_) { cudaFreeHost(arrival_host_a_); arrival_host_a_ = nullptr; arrival_a_ = nullptr; }
            if (arrival_host_b_) { cudaFreeHost(arrival_host_b_); arrival_host_b_ = nullptr; arrival_b_ = nullptr; }
            if (stall_host_) { cudaFreeHost(stall_host_); stall_host_ = nullptr; }
            stall_a_ = nullptr;
            stall_b_ = nullptr;
            if (base_host_) { cudaFreeHost(base_host_); base_host_ = nullptr; }
            if (base_dev_a_) {
                a_.bind_to_current_thread_noexcept();
                cudaFree(base_dev_a_);
                base_dev_a_ = nullptr;
            }
            if (base_dev_b_) {
                b_.bind_to_current_thread_noexcept();
                cudaFree(base_dev_b_);
                base_dev_b_ = nullptr;
            }
        }
    }
    if (in_kernel_available_) {
        watch_            = std::make_shared<ArWatchState>();
        watch_->id        = &ar_last_id_;
        watch_->arrival_a = static_cast<const unsigned char*>(arrival_host_a_);
        watch_->arrival_b = static_cast<const unsigned char*>(arrival_host_b_);
        start_ar_watchdog();
    }
}

DevicePair::~DevicePair() {
    stop_ar_watchdog();
    if (p2p_) {
        a_.bind_to_current_thread_noexcept();
        b_.bind_to_current_thread_noexcept();
        cudaDeviceDisablePeerAccess(b_.device);
        cudaDeviceDisablePeerAccess(a_.device);
    }
    if (host_a_) { cudaFreeHost(host_a_); }
    if (host_b_) { cudaFreeHost(host_b_); }
    if (arrival_host_a_) { cudaFreeHost(arrival_host_a_); }
    if (arrival_host_b_) { cudaFreeHost(arrival_host_b_); }
    if (stall_host_) { cudaFreeHost(stall_host_); }
    if (base_host_) { cudaFreeHost(base_host_); }
    if (base_dev_a_) {
        a_.bind_to_current_thread_noexcept();
        cudaFree(base_dev_a_);
    }
    if (base_dev_b_) {
        b_.bind_to_current_thread_noexcept();
        cudaFree(base_dev_b_);
    }
    if (small_host_a_) { cudaFreeHost(small_host_a_); }
    if (small_host_b_) { cudaFreeHost(small_host_b_); }
}

DevicePair::DevicePair(DevicePair&& other) noexcept
    : a_(std::move(other.a_)), b_(std::move(other.b_)), p2p_(other.p2p_),
      staging_(std::move(other.staging_)), in_kernel_available_(other.in_kernel_available_),
      in_kernel_bytes_(other.in_kernel_bytes_), host_a_(other.host_a_), host_b_(other.host_b_),
      dev_a_(other.dev_a_), dev_b_(other.dev_b_), arrival_a_(other.arrival_a_),
      arrival_b_(other.arrival_b_), arrival_host_a_(other.arrival_host_a_),
      arrival_host_b_(other.arrival_host_b_),
      stall_host_(other.stall_host_), stall_a_(other.stall_a_), stall_b_(other.stall_b_),
      ar_timeout_ns_(other.ar_timeout_ns_), ar_call_serial_(other.ar_call_serial_),
      ar_fault_skip_call_(other.ar_fault_skip_call_), base_host_(other.base_host_),
      base_dev_a_(other.base_dev_a_), base_dev_b_(other.base_dev_b_),
      ar_channels_(std::move(other.ar_channels_)), capturing_(other.capturing_),
      capture_channel_(other.capture_channel_), capture_index_(other.capture_index_),
      ar_eager_next_(other.ar_eager_next_), ar_last_id_(other.ar_last_id_.load()),
      small_available_(other.small_available_), small_host_a_(other.small_host_a_),
      small_host_b_(other.small_host_b_), small_dev_a_(other.small_dev_a_),
      small_dev_b_(other.small_dev_b_) {
    other.p2p_              = false;
    other.in_kernel_available_ = false;
    other.small_available_  = false;
    other.small_host_a_     = nullptr;
    other.small_host_b_     = nullptr;
    other.small_dev_a_      = nullptr;
    other.small_dev_b_      = nullptr;
    other.host_a_           = nullptr;
    other.host_b_           = nullptr;
    other.arrival_host_a_   = nullptr;
    other.arrival_host_b_   = nullptr;
    other.stall_host_       = nullptr;
    other.stall_a_          = nullptr;
    other.stall_b_          = nullptr;
    other.base_host_        = nullptr;
    other.base_dev_a_       = nullptr;
    other.base_dev_b_       = nullptr;
    other.capturing_        = false;
}

DevicePair& DevicePair::operator=(DevicePair&& other) noexcept {
    if (this == &other) { return *this; }
    a_ = std::move(other.a_);
    b_ = std::move(other.b_);
    p2p_ = other.p2p_;
    staging_ = std::move(other.staging_);
    in_kernel_available_ = other.in_kernel_available_;
    in_kernel_bytes_     = other.in_kernel_bytes_;
    host_a_              = other.host_a_;
    host_b_              = other.host_b_;
    dev_a_               = other.dev_a_;
    dev_b_               = other.dev_b_;
    arrival_a_           = other.arrival_a_;
    arrival_b_           = other.arrival_b_;
    arrival_host_a_      = other.arrival_host_a_;
    arrival_host_b_      = other.arrival_host_b_;
    stall_host_          = other.stall_host_;
    stall_a_             = other.stall_a_;
    stall_b_             = other.stall_b_;
    ar_timeout_ns_       = other.ar_timeout_ns_;
    ar_call_serial_      = other.ar_call_serial_;
    ar_fault_skip_call_  = other.ar_fault_skip_call_;
    base_host_           = other.base_host_;
    base_dev_a_          = other.base_dev_a_;
    base_dev_b_          = other.base_dev_b_;
    ar_channels_         = std::move(other.ar_channels_);
    capturing_           = other.capturing_;
    capture_channel_     = other.capture_channel_;
    capture_index_       = other.capture_index_;
    ar_eager_next_       = other.ar_eager_next_;
    ar_last_id_.store(other.ar_last_id_.load(), std::memory_order_relaxed);
    stop_ar_watchdog();
    watch_               = std::move(other.watch_);
    small_available_     = other.small_available_;
    small_host_a_        = other.small_host_a_;
    small_host_b_        = other.small_host_b_;
    small_dev_a_         = other.small_dev_a_;
    small_dev_b_         = other.small_dev_b_;
    other.p2p_              = false;
    other.in_kernel_available_ = false;
    other.host_a_           = nullptr;
    other.host_b_           = nullptr;
    other.arrival_host_a_   = nullptr;
    other.arrival_host_b_   = nullptr;
    other.stall_host_       = nullptr;
    other.stall_a_          = nullptr;
    other.stall_b_          = nullptr;
    other.base_host_        = nullptr;
    other.base_dev_a_       = nullptr;
    other.base_dev_b_       = nullptr;
    other.capturing_        = false;
    other.small_available_  = false;
    other.small_host_a_     = nullptr;
    other.small_host_b_     = nullptr;
    other.small_dev_a_      = nullptr;
    other.small_dev_b_      = nullptr;
    return *this;
}

void DevicePair::allreduce(void* data_a, void* data_b, std::size_t count_bytes,
                           cudaStream_t stream_a, cudaStream_t stream_b) {
    require_bytes(count_bytes);
    if (count_bytes == 0) { return; }
    // In-kernel path (WSL2, no P2P, mapped pinned available): both devices run
    // ar_exchange<true> on their compute streams. Each writes its delta to its own
    // mapped host buffer, signals an arrival id, spins on the peer's id,
    // then reads the peer's buffer and sums in place. No host synchronization -
    // the cross-GPU barrier is the kernel's arrival-id spin. The kernel is
    // ordered after the producing kernels on the same stream, and the caller's
    // next work (residual_add) is ordered after it, so no extra barrier is
    // needed. Falls through to host staging if mapped pinned is unavailable or
    // the delta exceeds the staging capacity.
    //
    // This transport is at the hardware floor for a batched prefill: the peer's PCIe link sets the
    // pace (on this host CUDA device 1 is Gen4 x4, ~7.9 GB/s counting both directions together,
    // against device 0's Gen5 x8 ~20 GB/s each way), and a [hidden, T] delta that must be written
    // and then read back costs 2 * hidden * T * 2 bytes per layer per device. Ordered slicing and
    // copy-engine transport were both measured against the plain single-block version; slicing
    // recovers the interleave loss (see ar_slices) while the copy engines were slower.
    if (in_kernel_available_ && count_bytes <= in_kernel_bytes_) {
        note_ar_call(count_bytes);
        const CollectiveId id = acquire_collective_id(stream_a, stream_b);
        const int count = static_cast<int>(count_bytes / 2);
        const auto address =
            reinterpret_cast<std::uintptr_t>(data_a) | reinterpret_cast<std::uintptr_t>(data_b) |
            reinterpret_cast<std::uintptr_t>(dev_a_) | reinterpret_cast<std::uintptr_t>(dev_b_);
        // require_bytes already bounds the payload to whole 16-byte groups; the pointer check only
        // decides whether the vectorized group loops may be used.
        const int groups = (address % 16 == 0) ? count / 8 : 0;
        auto* order_a = reinterpret_cast<unsigned long long*>(
            reinterpret_cast<char*>(arrival_a_) + kArSlotBytes);
        auto* order_b = reinterpret_cast<unsigned long long*>(
            reinterpret_cast<char*>(arrival_b_) + kArSlotBytes);
        if (small_available_ && count_bytes <= kArSmallBytes) {
            // Decode-sized: one block, one launch. About 128 calls ride each decode round, so the
            // launch the retired token bump used to add was pure latency at this payload. The thread
            // count stays at kArThreads: a smaller block regressed, because the payload's cost is
            // the mapped-host round trip and the group loops want the parallelism.
            const bool skip_peer = fault_skip_peer();
            a_.bind_to_current_thread();
            ar_exchange<true><<<1, kArThreads, 0, stream_a>>>(
                reinterpret_cast<const __nv_bfloat16*>(data_a),
                reinterpret_cast<__nv_bfloat16*>(data_a), static_cast<char*>(small_dev_a_),
                static_cast<const char*>(small_dev_b_), count, arrival_a_, arrival_b_, order_a,
                id.base_a, id.call_index, id.value, static_cast<int>(kArSmallBytes), groups,
                stall_a_, stall_b_, ar_timeout_ns_);
            if (!skip_peer) {
                b_.bind_to_current_thread();
                ar_exchange<true><<<1, kArThreads, 0, stream_b>>>(
                    reinterpret_cast<const __nv_bfloat16*>(data_b),
                    reinterpret_cast<__nv_bfloat16*>(data_b), static_cast<char*>(small_dev_b_),
                    static_cast<const char*>(small_dev_a_), count, arrival_b_, arrival_a_, order_b,
                    id.base_b, id.call_index, id.value, static_cast<int>(kArSmallBytes), groups,
                    stall_b_, stall_a_, ar_timeout_ns_);
            }
            return;
        }
        const int slices     = ar_slices(count_bytes);
        const int slot_bytes = static_cast<int>(kInKernelArBytes);
        const bool skip_peer = fault_skip_peer();
        a_.bind_to_current_thread();
        ar_exchange<true><<<slices, kArThreads, 0, stream_a>>>(
            reinterpret_cast<const __nv_bfloat16*>(data_a),
            reinterpret_cast<__nv_bfloat16*>(data_a), static_cast<char*>(dev_a_),
            static_cast<const char*>(dev_b_), count, arrival_a_, arrival_b_, order_a, id.base_a,
            id.call_index, id.value, slot_bytes, groups, stall_a_, stall_b_, ar_timeout_ns_);
        if (!skip_peer) {
            b_.bind_to_current_thread();
            ar_exchange<true><<<slices, kArThreads, 0, stream_b>>>(
                reinterpret_cast<const __nv_bfloat16*>(data_b),
                reinterpret_cast<__nv_bfloat16*>(data_b), static_cast<char*>(dev_b_),
                static_cast<const char*>(dev_a_), count, arrival_b_, arrival_a_, order_b, id.base_b,
                id.call_index, id.value, slot_bytes, groups, stall_b_, stall_a_, ar_timeout_ns_);
        }
        return;
    }
    if (p2p_) {
        a_.bind_to_current_thread();
        b_.bind_to_current_thread();
        // The deltas were produced on the caller's compute streams, which are
        // independent of the DevicePair's own peer-copy streams; settle them so
        // the peer copies read completed source halves.
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        CUDA_CHECK(cudaStreamSynchronize(stream_b));
        CUDA_CHECK(cudaMemcpyPeerAsync(data_a, a_.device, data_b, b_.device, count_bytes,
                                       a_.stream));
        CUDA_CHECK(cudaMemcpyPeerAsync(data_b, b_.device, data_a, a_.device, count_bytes,
                                       b_.stream));
        const std::size_t elements = count_bytes / 2;
        const std::size_t grid     = add_grid(elements);
        add_bf16_inplace<<<grid, kAddThreads, 0, a_.stream>>>(
            reinterpret_cast<__nv_bfloat16*>(data_a),
            reinterpret_cast<const __nv_bfloat16*>(data_b), elements);
        add_bf16_inplace<<<grid, kAddThreads, 0, b_.stream>>>(
            reinterpret_cast<__nv_bfloat16*>(data_b),
            reinterpret_cast<const __nv_bfloat16*>(data_a), elements);
        CUDA_CHECK(cudaStreamSynchronize(a_.stream));
        CUDA_CHECK(cudaStreamSynchronize(b_.stream));
        return;
    }
    // Host staging on the caller's compute streams: pinned buffer holds [a | b],
    // both halves count_bytes. The D2H copies are ordered after the kernels that
    // produced the deltas on those same streams, and the H2D copies are ordered
    // before the caller's next work (residual_add) on them, so only the two D2H
    // transfers need a barrier.
    const std::size_t total = 2 * count_bytes;
    if (!staging_ || staging_->size() < total) {
        staging_ = std::make_unique<PinnedHostBuffer>(total);
    }
    char* pinned = static_cast<char*>(staging_->data());
    // Bind before each copy: cudaMemcpyAsync orders on the stream's device, and
    // the driver validates the current device against the stream.
    a_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pinned, data_a, count_bytes, cudaMemcpyDeviceToHost, stream_a));
    b_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pinned + count_bytes, data_b, count_bytes,
                               cudaMemcpyDeviceToHost, stream_b));
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_b));
    const std::size_t elements = count_bytes / 2;
    const __nv_bfloat16* fa = reinterpret_cast<const __nv_bfloat16*>(pinned);
    const __nv_bfloat16* fb = reinterpret_cast<const __nv_bfloat16*>(pinned + count_bytes);
    __nv_bfloat16* sum      = reinterpret_cast<__nv_bfloat16*>(pinned);
    for (std::size_t i = 0; i < elements; ++i) {
        sum[i] = __hadd(fa[i], fb[i]);
    }
    a_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(data_a, pinned, count_bytes, cudaMemcpyHostToDevice, stream_a));
    b_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(data_b, pinned, count_bytes, cudaMemcpyHostToDevice, stream_b));
    // No H2D barrier: the caller enqueues residual_add on stream_a/stream_b
    // immediately after, so it is ordered after these transfers.
}

void DevicePair::sendrecv(const void* send_a, void* recv_a, const void* send_b, void* recv_b,
                          std::size_t count_bytes, cudaStream_t stream_a, cudaStream_t stream_b) {
    require_bytes(count_bytes);
    if (count_bytes == 0) { return; }
    // In-kernel transport: both devices run ar_exchange<false> on their compute streams. Each device
    // publishes its own send bytes to its mapped host staging, signals an arrival id, spins on the
    // peer's id, then copies the peer's staging slice into its receive buffer - no host
    // synchronization, and the same id/parity discipline as the all-reduce, so an allreduce and a
    // sendrecv may interleave within one round as long as both devices issue the same sequence.
    if (in_kernel_available_ && count_bytes <= in_kernel_bytes_) {
        note_ar_call(count_bytes);
        const CollectiveId id = acquire_collective_id(stream_a, stream_b);
        const int count = static_cast<int>(count_bytes / 2);
        const auto address = reinterpret_cast<std::uintptr_t>(send_a) |
                             reinterpret_cast<std::uintptr_t>(recv_a) |
                             reinterpret_cast<std::uintptr_t>(send_b) |
                             reinterpret_cast<std::uintptr_t>(recv_b) |
                             reinterpret_cast<std::uintptr_t>(dev_a_) |
                             reinterpret_cast<std::uintptr_t>(dev_b_);
        // require_bytes already bounds the payload to whole 16-byte groups; the pointer check only
        // decides whether the vectorized group loops may be used.
        const int groups = (address % 16 == 0) ? count / 8 : 0;
        auto* order_a = reinterpret_cast<unsigned long long*>(
            reinterpret_cast<char*>(arrival_a_) + kArSlotBytes);
        auto* order_b = reinterpret_cast<unsigned long long*>(
            reinterpret_cast<char*>(arrival_b_) + kArSlotBytes);
        if (small_available_ && count_bytes <= kArSmallBytes) {
            const bool skip_peer = fault_skip_peer();
            a_.bind_to_current_thread();
            ar_exchange<false><<<1, kArThreads, 0, stream_a>>>(
                static_cast<const __nv_bfloat16*>(send_a), static_cast<__nv_bfloat16*>(recv_a),
                static_cast<char*>(small_dev_a_), static_cast<const char*>(small_dev_b_), count,
                arrival_a_, arrival_b_, order_a, id.base_a, id.call_index, id.value,
                static_cast<int>(kArSmallBytes), groups, stall_a_, stall_b_, ar_timeout_ns_);
            if (!skip_peer) {
                b_.bind_to_current_thread();
                ar_exchange<false><<<1, kArThreads, 0, stream_b>>>(
                    static_cast<const __nv_bfloat16*>(send_b), static_cast<__nv_bfloat16*>(recv_b),
                    static_cast<char*>(small_dev_b_), static_cast<const char*>(small_dev_a_), count,
                    arrival_b_, arrival_a_, order_b, id.base_b, id.call_index, id.value,
                    static_cast<int>(kArSmallBytes), groups, stall_b_, stall_a_, ar_timeout_ns_);
            }
            return;
        }
        const int slices     = ar_slices(count_bytes);
        const int slot_bytes = static_cast<int>(kInKernelArBytes);
        const bool skip_peer = fault_skip_peer();
        a_.bind_to_current_thread();
        ar_exchange<false><<<slices, kArThreads, 0, stream_a>>>(
            static_cast<const __nv_bfloat16*>(send_a), static_cast<__nv_bfloat16*>(recv_a),
            static_cast<char*>(dev_a_), static_cast<const char*>(dev_b_), count, arrival_a_,
            arrival_b_, order_a, id.base_a, id.call_index, id.value, slot_bytes, groups, stall_a_,
            stall_b_, ar_timeout_ns_);
        if (!skip_peer) {
            b_.bind_to_current_thread();
            ar_exchange<false><<<slices, kArThreads, 0, stream_b>>>(
                static_cast<const __nv_bfloat16*>(send_b), static_cast<__nv_bfloat16*>(recv_b),
                static_cast<char*>(dev_b_), static_cast<const char*>(dev_a_), count, arrival_b_,
                arrival_a_, order_b, id.base_b, id.call_index, id.value, slot_bytes, groups,
                stall_b_, stall_a_, ar_timeout_ns_);
        }
        return;
    }
    if (p2p_) {
        a_.bind_to_current_thread();
        b_.bind_to_current_thread();
        // The send halves were produced on the caller's compute streams, which are independent of the
        // DevicePair's own peer-copy streams; settle them so every copy reads a completed source half.
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        CUDA_CHECK(cudaStreamSynchronize(stream_b));
        CUDA_CHECK(cudaMemcpyPeerAsync(recv_a, a_.device, send_b, b_.device, count_bytes, a_.stream));
        CUDA_CHECK(cudaMemcpyPeerAsync(recv_b, b_.device, send_a, a_.device, count_bytes, b_.stream));
        CUDA_CHECK(cudaStreamSynchronize(a_.stream));
        CUDA_CHECK(cudaStreamSynchronize(b_.stream));
        return;
    }
    // Host staging: the pinned buffer holds [a's send | b's send] and each receive copies the other
    // half back. Only the two D2H transfers need a barrier, exactly as the all-reduce fallback does.
    const std::size_t total = 2 * count_bytes;
    if (!staging_ || staging_->size() < total) {
        staging_ = std::make_unique<PinnedHostBuffer>(total);
    }
    char* pinned = static_cast<char*>(staging_->data());
    a_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pinned, send_a, count_bytes, cudaMemcpyDeviceToHost, stream_a));
    b_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pinned + count_bytes, send_b, count_bytes, cudaMemcpyDeviceToHost,
                               stream_b));
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_b));
    a_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(recv_a, pinned + count_bytes, count_bytes, cudaMemcpyHostToDevice,
                               stream_a));
    b_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(recv_b, pinned, count_bytes, cudaMemcpyHostToDevice, stream_b));
}

ShardedTensor shard_view(const Tensor& full, int dim, std::int32_t local_len, int shard) {
    if (dim < 0 || dim >= 4) { throw std::invalid_argument("tp shard_view: dim out of range"); }
    if (shard < 0 || local_len <= 0) {
        throw std::invalid_argument("tp shard_view: invalid shard geometry");
    }
    const std::int32_t full_len = full.ne[dim];
    if (local_len > full_len) {
        throw std::invalid_argument("tp shard_view: local_len exceeds full extent");
    }
    ShardedTensor out;
    out.shard  = shard;
    out.shards = 1;
    out.local  = full.slice(dim, 0, local_len);
    return out;
}

ShardedTensor column_shard(const Tensor& weight, int shard, int shards) {
    if (shards < 1 || shard < 0 || shard >= shards) {
        throw std::invalid_argument("tp column_shard: invalid shard index");
    }
    if (weight.ne[3] != 1 || weight.ne[2] != 1) {
        throw std::invalid_argument("tp column_shard: expects a rank-2 weight");
    }
    const std::int32_t cols      = weight.ne[1];
    const std::int32_t base      = cols / shards;
    const std::int32_t remainder = cols % shards;
    const std::int32_t local     = base + (shard < remainder ? 1 : 0);
    const std::int32_t offset    = shard * base + std::min(shard, remainder);
    ShardedTensor out;
    out.shard  = shard;
    out.shards = shards;
    out.local  = weight.slice(1, offset, local);
    return out;
}

ShardedTensor row_shard(const Tensor& weight, int shard, int shards) {
    if (shards < 1 || shard < 0 || shard >= shards) {
        throw std::invalid_argument("tp row_shard: invalid shard index");
    }
    if (weight.ne[3] != 1 || weight.ne[2] != 1) {
        throw std::invalid_argument("tp row_shard: expects a rank-2 weight");
    }
    const std::int32_t rows      = weight.ne[0];
    const std::int32_t base      = rows / shards;
    const std::int32_t remainder = rows % shards;
    const std::int32_t local     = base + (shard < remainder ? 1 : 0);
    const std::int32_t offset    = shard * base + std::min(shard, remainder);
    ShardedTensor out;
    out.shard  = shard;
    out.shards = shards;
    out.local  = weight.slice(0, offset, local);
    return out;
}

} // namespace ninfer::tp
