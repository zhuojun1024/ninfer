// Can the TP-2 in-kernel allreduce co-run with a GPU-filling compute kernel?
// Reproduces the production protocol (ordered slice writes to mapped pinned host staging, arrival id,
// spin on the peer id, read-and-add) on two RTX 5060 Ti and runs it on a second stream while a
// compute kernel occupies the SMs. Reports the collective's start offset against the compute end
// (starvation) and how much the collective's own transfer inflates while the compute runs.
//   nvcc -O3 -arch=sm_120a -o bench_ar_overlap.exe bench_ar_overlap.cu
//   bench_ar_overlap.exe [dev_a] [dev_b] [calls] [target_ms] [blocks_per_sm] [chain] [style]
//   style 0 = FP32-FMA bound, 1 = HBM streaming bound
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,           \
                         cudaGetErrorString(e_));                                                  \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)
__global__ void ar_protocol(uint4* __restrict__ mine_dev, uint4* __restrict__ mine_host,
                            uint4* __restrict__ peer_host, unsigned long long* order,
                            unsigned long long* arrival_mine,
                            const unsigned long long* __restrict__ arrival_peer, int units,
                            int slice_units, unsigned long long token, int pipeline) {
    const int b = blockIdx.x;
    if (pipeline != 0) {
        if (threadIdx.x == 0 && b > 0) {
            while (*(volatile unsigned long long*)&order[b - 1] != token) { __nanosleep(64); }
        }
        __syncthreads();
    }
    const int lo = b * slice_units;
    const int hi = min(lo + slice_units, units);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) { mine_host[i] = mine_dev[i]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile unsigned long long*)&order[b]        = token;
        *(volatile unsigned long long*)&arrival_mine[b] = token;
    }
    if (threadIdx.x == 0) {
        while (*(const volatile unsigned long long*)&arrival_peer[b] != token) { __nanosleep(64); }
    }
    __syncthreads();
    __threadfence_system();
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const uint4 p = peer_host[i];
        uint4       m = mine_dev[i];
        m.x += p.x; m.y += p.y; m.z += p.z; m.w += p.w;
        mine_dev[i] = m;
    }
}
__global__ void compute_burn(float* __restrict__ sink, int iters, float seed) {
    float c0 = seed + threadIdx.x;
    float c1 = seed + threadIdx.x + 1.0f;
    float c2 = seed + threadIdx.x + 2.0f;
    float c3 = seed + threadIdx.x + 3.0f;
    const float k = 1.0000001f;
    for (int i = 0; i < iters; ++i) {
        c0 = fmaf(c0, k, 1e-7f);
        c1 = fmaf(c1, k, 1e-7f);
        c2 = fmaf(c2, k, 1e-7f);
        c3 = fmaf(c3, k, 1e-7f);
    }
    const float r = c0 + c1 + c2 + c3;
    if (r == 1.0e30f) { sink[blockIdx.x * blockDim.x + threadIdx.x] = r; }
}
__global__ void stream_burn(float* __restrict__ buf, long long n, int passes, float seed) {
    long long       i      = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    float           acc    = seed;
    for (int k = 0; k < passes; ++k) {
        for (long long j = i; j < n; j += stride) { acc += buf[j]; }
    }
    if (acc == 1.0e30f) { buf[i % n] = acc; }
}
struct Buf {
    uint4*              dev     = nullptr;
    uint4*              host    = nullptr;
    unsigned long long* order   = nullptr;
    unsigned long long* arrival = nullptr;
    float*              sink    = nullptr;
    float*              stream  = nullptr;
    long long           stream_n = 0;
    int                 device  = 0;
};

static Buf make_buf(int device, std::size_t bytes) {
    Buf b;
    b.device = device;
    CK(cudaSetDevice(device));
    CK(cudaMalloc(&b.dev, bytes));
    void* h = nullptr;
    CK(cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    b.host = (uint4*)h;
    void* o = nullptr;
    CK(cudaHostAlloc(&o, 4096, cudaHostAllocMapped | cudaHostAllocPortable));
    b.order = (unsigned long long*)o;
    void* a = nullptr;
    CK(cudaHostAlloc(&a, 4096, cudaHostAllocMapped | cudaHostAllocPortable));
    b.arrival = (unsigned long long*)a;
    CK(cudaMemset(b.dev, 1, bytes));
    CK(cudaMemset(b.order, 0, 4096));
    CK(cudaMemset(b.arrival, 0, 4096));
    CK(cudaMalloc(&b.sink, 1 << 20));
    b.stream_n = (long long)(128u << 20) / 4;
    CK(cudaMalloc(&b.stream, (std::size_t)b.stream_n * sizeof(float)));
    CK(cudaMemset(b.stream, 0, (std::size_t)b.stream_n * sizeof(float)));
    return b;
}
static void free_buf(Buf& b) {
    CK(cudaSetDevice(b.device));
    cudaFree(b.dev);
    cudaFreeHost(b.host);
    cudaFreeHost(b.order);
    cudaFreeHost(b.arrival);
    cudaFree(b.sink);
    cudaFree(b.stream);
}
static int ar_slices_of(std::size_t bytes) {
    if (bytes <= (256ULL << 10)) { return 1; }
    const std::size_t want = (bytes + (512ULL << 10) - 1) / (512ULL << 10);
    return (int)std::min<std::size_t>(std::max<std::size_t>(want, 2), 8);
}
struct Ev {
    cudaEvent_t e = nullptr;
    explicit Ev(int device) { CK(cudaSetDevice(device)); cudaEventCreate(&e); }
    ~Ev() { cudaEventDestroy(e); }
};

static double ar_alone(Buf& A, Buf& B, cudaStream_t sa, cudaStream_t sb, int slices, int units,
                       int slice_units, int calls, unsigned long long& token) {
    Ev a0(0), a1(0), b0(1), b1(1);
    CK(cudaSetDevice(0)); cudaEventRecord(a0.e, sa);
    CK(cudaSetDevice(1)); cudaEventRecord(b0.e, sb);
    for (int i = 0; i < calls; ++i) {
        ++token;
        CK(cudaSetDevice(0));
        ar_protocol<<<slices, 1024, 0, sa>>>(A.dev, A.host, B.host, A.order, A.arrival, B.arrival,
                                             units, slice_units, token, 1);
        CK(cudaSetDevice(1));
        ar_protocol<<<slices, 1024, 0, sb>>>(B.dev, B.host, A.host, B.order, B.arrival, A.arrival,
                                             units, slice_units, token, 1);
    }
    CK(cudaSetDevice(0)); cudaEventRecord(a1.e, sa);
    CK(cudaSetDevice(1)); cudaEventRecord(b1.e, sb);
    CK(cudaSetDevice(0)); cudaEventSynchronize(a1.e);
    CK(cudaSetDevice(1)); cudaEventSynchronize(b1.e);
    float ma = 0.0f, mb = 0.0f;
    CK(cudaSetDevice(0)); cudaEventElapsedTime(&ma, a0.e, a1.e);
    CK(cudaSetDevice(1)); cudaEventElapsedTime(&mb, b0.e, b1.e);
    return ma > mb ? ma : mb;
}
static void launch_one(Buf& X, cudaStream_t s, int style, int blocks, int iters) {
    if (style == 1) {
        stream_burn<<<blocks, 256, 0, s>>>(X.stream, X.stream_n, iters, 1.0f);
    } else {
        compute_burn<<<blocks, 256, 0, s>>>(X.sink, iters, 2.0f);
    }
}
static double one_compute(int device, Buf& X, cudaStream_t s, int style, int blocks, int iters) {
    Ev c0(device), c1(device);
    CK(cudaSetDevice(device));
    cudaEventRecord(c0.e, s);
    launch_one(X, s, style, blocks, iters);
    cudaEventRecord(c1.e, s);
    CK(cudaStreamSynchronize(s));
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, c0.e, c1.e);
    return ms;
}

int main(int argc, char** argv) {
    const int dev_a = argc > 1 ? std::atoi(argv[1]) : 0;
    const int dev_b = argc > 2 ? std::atoi(argv[2]) : 1;
    const int calls = argc > 3 ? std::atoi(argv[3]) : 16;
    const double target_ms = argc > 4 ? std::atof(argv[4]) : 60.0;
    const int bps = argc > 5 ? std::atoi(argv[5]) : 8;
    const int chain = argc > 6 ? std::atoi(argv[6]) : 1;
    const int style = argc > 7 ? std::atoi(argv[7]) : 0;
    int sm_a = 0;
    CK(cudaSetDevice(dev_a));
    CK(cudaDeviceGetAttribute(&sm_a, cudaDevAttrMultiProcessorCount, dev_a));
    cudaStream_t sc_a, sx_a, sc_b, sx_b;
    CK(cudaStreamCreateWithFlags(&sc_a, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&sx_a, cudaStreamNonBlocking));
    CK(cudaSetDevice(dev_b));
    CK(cudaStreamCreateWithFlags(&sc_b, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&sx_b, cudaStreamNonBlocking));
    std::printf("devices a=%d(%d SM) b=%d calls=%d target=%.1f ms bps=%d chain=%d style=%d\n",
                dev_a, sm_a, dev_b, calls, target_ms, bps, chain, style);
    const std::vector<std::size_t> payloads = {(std::size_t)5120 * 512 * 2, (std::size_t)5120 * 1024 * 2};
    for (std::size_t bytes : payloads) {
        Buf A = make_buf(dev_a, bytes), B = make_buf(dev_b, bytes);
        const int units       = (int)(bytes / 16);
        const int slices      = ar_slices_of(bytes);
        const int slice_units = (units + slices - 1) / slices;
        unsigned long long token = 1000;
        const double t_ar = ar_alone(A, B, sx_a, sx_b, slices, units, slice_units, calls, token);
        const int blocks = sm_a * bps;
        const int cal = style == 1 ? 8 : 20000;
        const double single = one_compute(dev_a, A, sc_a, style, blocks, cal);
        const int iters = single > 0.001 ? (int)std::max(200.0, cal * (target_ms / single)) : cal;
        const int citer = std::max(iters / chain, 1);
        std::printf("\npayload %8zu B slices=%d | collective alone %8.3f ms (%6.3f ms/call) | calibration %d iters=%7.3f ms -> %d kernels x %d iters\n",
                    bytes, slices, t_ar, t_ar / calls, cal, single, chain, citer);
        Ev ca0(dev_a), ca1(dev_a), xa0(dev_a), xa1(dev_a);
        CK(cudaSetDevice(dev_a));
        cudaEventRecord(ca0.e, sc_a);
        for (int k = 0; k < chain; ++k) { launch_one(A, sc_a, style, blocks, citer); }
        cudaEventRecord(ca1.e, sc_a);
        cudaEventRecord(xa0.e, sx_a);
        CK(cudaSetDevice(dev_b));
        for (int k = 0; k < chain; ++k) { launch_one(B, sc_b, style, blocks, citer); }
        for (int i = 0; i < calls; ++i) {
            ++token;
            CK(cudaSetDevice(dev_a));
            ar_protocol<<<slices, 1024, 0, sx_a>>>(A.dev, A.host, B.host, A.order, A.arrival,
                                                   B.arrival, units, slice_units, token, 1);
            CK(cudaSetDevice(dev_b));
            ar_protocol<<<slices, 1024, 0, sx_b>>>(B.dev, B.host, A.host, B.order, B.arrival,
                                                   A.arrival, units, slice_units, token, 1);
        }
        CK(cudaSetDevice(dev_a));
        cudaEventRecord(xa1.e, sx_a);
        CK(cudaEventSynchronize(xa1.e));
        CK(cudaEventSynchronize(ca1.e));
        float comp = 0.0f, ar = 0.0f, off = 0.0f, cend = 0.0f, xend = 0.0f;
        cudaEventElapsedTime(&comp, ca0.e, ca1.e);
        cudaEventElapsedTime(&ar, xa0.e, xa1.e);
        cudaEventElapsedTime(&off, ca0.e, xa0.e);
        cudaEventElapsedTime(&cend, ca0.e, ca1.e);
        cudaEventElapsedTime(&xend, ca0.e, xa1.e);
        std::printf("   concurrent: compute %8.3f | collective %8.3f | ar_start_offset %8.3f | ar_end %8.3f | ar alone %8.3f | ar_inflate %5.1f%%\n",
                    comp, ar, off, xend, t_ar, 100.0 * (ar - t_ar) / t_ar);
        std::printf("   verdict: %s\n",
                    off >= comp ? "STARVED (collective started only after compute ended)"
                                : (xend <= cend ? "FULL OVERLAP (collective finished before compute)"
                                                : "PARTIAL (collective started during compute)"));
        free_buf(A);
        free_buf(B);
    }
    return 0;
}
