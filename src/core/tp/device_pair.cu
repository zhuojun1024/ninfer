#include "core/tp/device_pair.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
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
template <bool kAdd>
__global__ void ar_exchange(const __nv_bfloat16* local, __nv_bfloat16* out,
                            char* host_mine_base, const char* host_other_base, int count,
                            int* arrival_mine, int* arrival_other, int* order_mine,
                            int* token_ptr, int slot_bytes, int groups, int fuse_bump) {
    // The token lives in mapped host memory (either device may run this call), where a per-thread
    // load would be a system-scope read per thread. One read per block is enough: the value is fixed
    // for the whole call, so publish it through shared memory. The single-block decode-sized call
    // (fuse_bump) advances the counter from that same thread instead of running bump_ar_token first:
    // a second launch is pure latency at this payload, and only this device reads its own counter.
    __shared__ int shared_token;
    if (threadIdx.x == 0) {
        if (fuse_bump != 0) {
            const int next            = *(volatile int*)token_ptr + 1;
            *(volatile int*)token_ptr = next;
            __threadfence_system();
            shared_token = next;
        } else {
            shared_token = *(const volatile int*)token_ptr;
        }
    }
    __syncthreads();
    const int token  = shared_token;
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
        while (*(const volatile int*)(order_mine + blockIdx.x - 1) != token) { __nanosleep(100); }
    }
    if (blocks > 1) { __syncthreads(); }
    for (int i = begin + threadIdx.x; i < end; i += stride) { mine_groups[i] = local_groups[i]; }
    if (blockIdx.x == 0) {
        for (int i = tail + threadIdx.x; i < count; i += stride) { host_mine[i] = local[i]; }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile int*)(order_mine + blockIdx.x)  = token;
        *(volatile int*)(arrival_mine + blockIdx.x) = token;
        __threadfence_system();
        while (*(const volatile int*)(arrival_other + blockIdx.x) != token) { __nanosleep(100); }
    }
    __syncthreads();
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
constexpr std::size_t kArSlotBytes     = 128; // one cache line per slot array (kArMaxSlices ints)
constexpr std::size_t kArTokenBytes    = 2 * kArSlotBytes; // arrival array, then write-order chain

// Decode-sized staging for the size-keyed transport. A single token's [hidden, 1] delta is a few
// KiB: the 24 MiB prefill slots are almost all waste, and sharing them would let a small call's
// parity slot alias a prefill payload the peer is still reading. This buffer holds one small
// payload per parity instead.
constexpr std::size_t kArSmallBytes = 64ULL << 10; // staging per device, per parity

// Advances one device's arrival token. Launched immediately before the allreduce kernel on the
// same stream, so the allreduce's device read sees the new value. Both devices run one bump per
// allreduce call from the same starting value, which keeps their tokens equal at every call.
__global__ void bump_ar_token(int* token) { *token = *token + 1; }

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

DevicePair::DevicePair(int device_a, int device_b) : a_(device_a), b_(device_b) {
    if (device_a == device_b) {
        throw std::invalid_argument("tp DevicePair: devices must be distinct");
    }
    const char* ar_strategy = std::getenv("NINFER_TP2_AR_STRATEGY");
    // NINFER_TP2_AR_STRATEGY=kernel pins every payload to the sliced transport, which is the A/B
    // reference for the size-keyed policy.
    ar_size_keyed_ = ar_strategy == nullptr || std::strcmp(ar_strategy, "kernel") != 0;
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
                        // The tokens are host memory mapped into both devices, so whichever shard
                        // drives the pair may bump and read them.
                        bool tokens = false;
                        if (cudaHostAlloc(reinterpret_cast<void**>(&token_host_), 2 * sizeof(int),
                                          cudaHostAllocPortable | cudaHostAllocMapped) ==
                            cudaSuccess) {
                            tokens = cudaHostGetDevicePointer(reinterpret_cast<void**>(&token_a_),
                                                              token_host_, 0) == cudaSuccess;
                            if (tokens) {
                                token_host_[0] = 0;
                                token_host_[1] = 0;
                                token_b_       = token_a_ + 1;
                            }
                        }
                        if (tokens) {
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
        if (in_kernel_available_ && ar_size_keyed_) {
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
            if (token_host_) { cudaFreeHost(token_host_); token_host_ = nullptr; }
            token_a_ = nullptr;
            token_b_ = nullptr;
        }
    }
}

DevicePair::~DevicePair() {
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
    if (token_host_) { cudaFreeHost(token_host_); }
    if (small_host_a_) { cudaFreeHost(small_host_a_); }
    if (small_host_b_) { cudaFreeHost(small_host_b_); }
}

DevicePair::DevicePair(DevicePair&& other) noexcept
    : a_(std::move(other.a_)), b_(std::move(other.b_)), p2p_(other.p2p_),
      staging_(std::move(other.staging_)), in_kernel_available_(other.in_kernel_available_),
      in_kernel_bytes_(other.in_kernel_bytes_), host_a_(other.host_a_), host_b_(other.host_b_),
      dev_a_(other.dev_a_), dev_b_(other.dev_b_), arrival_a_(other.arrival_a_),
      arrival_b_(other.arrival_b_), arrival_host_a_(other.arrival_host_a_),
      arrival_host_b_(other.arrival_host_b_), token_host_(other.token_host_),
      token_a_(other.token_a_), token_b_(other.token_b_), ar_size_keyed_(other.ar_size_keyed_),
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
    other.token_host_       = nullptr;
    other.token_a_          = nullptr;
    other.token_b_          = nullptr;
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
    token_host_          = other.token_host_;
    token_a_             = other.token_a_;
    token_b_             = other.token_b_;
    ar_size_keyed_       = other.ar_size_keyed_;
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
    other.token_host_       = nullptr;
    other.token_a_          = nullptr;
    other.token_b_          = nullptr;
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
    // mapped host buffer, signals an arrival token, spins on the peer's token,
    // then reads the peer's buffer and sums in place. No host synchronization -
    // the cross-GPU barrier is the kernel's arrival-token spin. The kernel is
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
        const int count = static_cast<int>(count_bytes / 2);
        const auto address =
            reinterpret_cast<std::uintptr_t>(data_a) | reinterpret_cast<std::uintptr_t>(data_b) |
            reinterpret_cast<std::uintptr_t>(dev_a_) | reinterpret_cast<std::uintptr_t>(dev_b_);
        // require_bytes already bounds the payload to whole 16-byte groups; the pointer check only
        // decides whether the vectorized group loops may be used.
        const int groups = (address % 16 == 0) ? count / 8 : 0;
        int*      order_a = reinterpret_cast<int*>(reinterpret_cast<char*>(arrival_a_) + kArSlotBytes);
        int*      order_b = reinterpret_cast<int*>(reinterpret_cast<char*>(arrival_b_) + kArSlotBytes);
        if (ar_size_keyed_ && small_available_ && count_bytes <= kArSmallBytes) {
            // Decode-sized: one block, one launch. Folding the token bump into the kernel takes one
            // launch off every call, and 128 calls ride each decode round. The thread count stays at
            // kArThreads: a smaller block regressed, because the payload's cost is the mapped-host
            // round trip and the group loops want the parallelism.
            a_.bind_to_current_thread();
            ar_exchange<true><<<1, kArThreads, 0, stream_a>>>(
                reinterpret_cast<const __nv_bfloat16*>(data_a),
                reinterpret_cast<__nv_bfloat16*>(data_a), static_cast<char*>(small_dev_a_),
                static_cast<const char*>(small_dev_b_), count, arrival_a_, arrival_b_, order_a,
                token_a_, static_cast<int>(kArSmallBytes), groups, 1);
            b_.bind_to_current_thread();
            ar_exchange<true><<<1, kArThreads, 0, stream_b>>>(
                reinterpret_cast<const __nv_bfloat16*>(data_b),
                reinterpret_cast<__nv_bfloat16*>(data_b), static_cast<char*>(small_dev_b_),
                static_cast<const char*>(small_dev_a_), count, arrival_b_, arrival_a_, order_b,
                token_b_, static_cast<int>(kArSmallBytes), groups, 1);
            return;
        }
        const int slices     = ar_slices(count_bytes);
        const int slot_bytes = static_cast<int>(kInKernelArBytes);
        // Each device advances its own token before running the kernel, and the kernel derives its
        // double-buffer parity from that value, so a captured sequence needs no host-side counter.
        a_.bind_to_current_thread();
        bump_ar_token<<<1, 1, 0, stream_a>>>(token_a_);
        ar_exchange<true><<<slices, kArThreads, 0, stream_a>>>(
            reinterpret_cast<const __nv_bfloat16*>(data_a),
            reinterpret_cast<__nv_bfloat16*>(data_a), static_cast<char*>(dev_a_),
            static_cast<const char*>(dev_b_), count, arrival_a_, arrival_b_, order_a, token_a_,
            slot_bytes, groups, 0);
        b_.bind_to_current_thread();
        bump_ar_token<<<1, 1, 0, stream_b>>>(token_b_);
        ar_exchange<true><<<slices, kArThreads, 0, stream_b>>>(
            reinterpret_cast<const __nv_bfloat16*>(data_b),
            reinterpret_cast<__nv_bfloat16*>(data_b), static_cast<char*>(dev_b_),
            static_cast<const char*>(dev_a_), count, arrival_b_, arrival_a_, order_b, token_b_,
            slot_bytes, groups, 0);
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
    // publishes its own send bytes to its mapped host staging, signals an arrival token, spins on the
    // peer's token, then copies the peer's staging slice into its receive buffer - no host
    // synchronization, and the same token/parity discipline as the all-reduce, so an allreduce and a
    // sendrecv may interleave within one round as long as both devices issue the same sequence.
    if (in_kernel_available_ && count_bytes <= in_kernel_bytes_) {
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
        int*      order_a = reinterpret_cast<int*>(reinterpret_cast<char*>(arrival_a_) + kArSlotBytes);
        int*      order_b = reinterpret_cast<int*>(reinterpret_cast<char*>(arrival_b_) + kArSlotBytes);
        if (ar_size_keyed_ && small_available_ && count_bytes <= kArSmallBytes) {
            a_.bind_to_current_thread();
            ar_exchange<false><<<1, kArThreads, 0, stream_a>>>(
                static_cast<const __nv_bfloat16*>(send_a), static_cast<__nv_bfloat16*>(recv_a),
                static_cast<char*>(small_dev_a_), static_cast<const char*>(small_dev_b_), count,
                arrival_a_, arrival_b_, order_a, token_a_, static_cast<int>(kArSmallBytes), groups, 1);
            b_.bind_to_current_thread();
            ar_exchange<false><<<1, kArThreads, 0, stream_b>>>(
                static_cast<const __nv_bfloat16*>(send_b), static_cast<__nv_bfloat16*>(recv_b),
                static_cast<char*>(small_dev_b_), static_cast<const char*>(small_dev_a_), count,
                arrival_b_, arrival_a_, order_b, token_b_, static_cast<int>(kArSmallBytes), groups, 1);
            return;
        }
        const int slices     = ar_slices(count_bytes);
        const int slot_bytes = static_cast<int>(kInKernelArBytes);
        a_.bind_to_current_thread();
        bump_ar_token<<<1, 1, 0, stream_a>>>(token_a_);
        ar_exchange<false><<<slices, kArThreads, 0, stream_a>>>(
            static_cast<const __nv_bfloat16*>(send_a), static_cast<__nv_bfloat16*>(recv_a),
            static_cast<char*>(dev_a_), static_cast<const char*>(dev_b_), count, arrival_a_,
            arrival_b_, order_a, token_a_, slot_bytes, groups, 0);
        b_.bind_to_current_thread();
        bump_ar_token<<<1, 1, 0, stream_b>>>(token_b_);
        ar_exchange<false><<<slices, kArThreads, 0, stream_b>>>(
            static_cast<const __nv_bfloat16*>(send_b), static_cast<__nv_bfloat16*>(recv_b),
            static_cast<char*>(dev_b_), static_cast<const char*>(dev_a_), count, arrival_b_,
            arrival_a_, order_b, token_b_, slot_bytes, groups, 0);
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
