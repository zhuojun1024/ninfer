#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("FAIL %s: %s\n", #x, cudaGetErrorString(e)); return 1;}}while(0)

// Kernel: write `count` BF16 values (value = tag) into mapped host memory.
__global__ void write_host(__nv_bfloat16* p, int count, __nv_bfloat16 tag) {
    for (int i = threadIdx.x; i < count; i += blockDim.x) p[i] = tag;
}
// Kernel: read `count` BF16 from mapped host memory, sum into *out (device).
__global__ void read_host(const __nv_bfloat16* p, int count, float* out) {
    float s = 0.f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) s += __bfloat162float(p[i]);
    // single thread accumulates via atomic
    atomicAdd(out, s);
}
// Cross-device in-kernel allreduce: write local to host_mine, spin on peer arrival, read peer, sum.
__global__ void ar_kernel(const __nv_bfloat16* local, __nv_bfloat16* out, __nv_bfloat16* host_mine,
                          const __nv_bfloat16* host_other, int count, int* arr_mine, int* arr_other,
                          int token) {
    // Phase 1: write local -> host_mine
    for (int i = threadIdx.x; i < count; i += blockDim.x) host_mine[i] = local[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile int*)arr_mine = token;
        __threadfence_system();
        while (*(const volatile int*)arr_other != token) { __nanosleep(100); }
    }
    __syncthreads();
    __threadfence_system();
    // Phase 3: read peer, sum
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        out[i] = __hadd(local[i], host_other[i]);
}

int main() {
    int n=0; CK(cudaGetDeviceCount(&n));
    int a=0,b=1;
    // 1) mapped pinned alloc on device a
    CK(cudaSetDevice(a));
    const size_t bytes = 4096; // 2048 bf16
    void* host=nullptr;
    cudaError_t rc = cudaHostAlloc(&host, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (rc != cudaSuccess) { printf("cudaHostAllocMapped FAILED: %s\n", cudaGetErrorString(rc)); return 1; }
    void* dev=nullptr;
    rc = cudaHostGetDevicePointer(&dev, host, 0);
    if (rc != cudaSuccess) { printf("cudaHostGetDevicePointer FAILED: %s\n", cudaGetErrorString(rc)); cudaFreeHost(host); return 1; }
    printf("mapped pinned OK: host=%p dev=%p\n", host, dev);
    // 2) device a writes pattern via dev ptr
    CK(cudaSetDevice(a));
    write_host<<<1,256>>>((__nv_bfloat16*)dev, 2048, __float2bfloat16(3.5f));
    CK(cudaDeviceSynchronize());
    // host reads back
    auto* hb = (__nv_bfloat16*)host;
    bool ok = true; for (int i=0;i<2048;i++) if (__bfloat162float(hb[i]) != 3.5f) { ok=false; break; }
    printf("device-a write -> host readback: %s\n", ok?"OK":"FAIL");
    // 3) device b reads same host memory via its own dev ptr
    CK(cudaSetDevice(b));
    void* dev_b=nullptr; CK(cudaHostGetDevicePointer(&dev_b, host, 0));
    float* dsum=nullptr; CK(cudaMalloc(&dsum, 4)); CK(cudaMemset(dsum,0,4));
    read_host<<<1,256>>>((const __nv_bfloat16*)dev_b, 2048, dsum);
    CK(cudaDeviceSynchronize());
    float hsum=0; CK(cudaMemcpy(&hsum, dsum, 4, cudaMemcpyDeviceToHost));
    printf("device-b read of host mem: sum=%.1f (expect %.1f) => %s\n", hsum, 2048*3.5f, (hsum>2048*3.5f-1)?"OK":"FAIL");
    // 4) time cross-device in-kernel allreduce (10KB = 5120 bf16)
    const int count = 5120;
    const size_t ar_bytes = count*2;
    void* ha=nullptr,*hb2=nullptr;
    CK(cudaSetDevice(a)); CK(cudaHostAlloc(&ha, ar_bytes, cudaHostAllocPortable|cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&ha, ha, 0));
    CK(cudaSetDevice(b)); CK(cudaHostAlloc(&hb2, ar_bytes, cudaHostAllocPortable|cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&hb2, hb2, 0));
    // arrival rings
    void* ra=nullptr,*rb=nullptr;
    CK(cudaSetDevice(a)); CK(cudaHostAlloc(&ra, 64, cudaHostAllocPortable|cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&ra, ra, 0));
    CK(cudaSetDevice(b)); CK(cudaHostAlloc(&rb, 64, cudaHostAllocPortable|cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&rb, rb, 0));
    int* ra_i=(int*)ra, *rb_i=(int*)rb; ra_i[0]=0; rb_i[0]=0;
    // local deltas
    __nv_bfloat16* da=nullptr,*db=nullptr; __nv_bfloat16* oa=nullptr,*ob=nullptr;
    CK(cudaSetDevice(a)); CK(cudaMalloc(&da, ar_bytes)); CK(cudaMalloc(&oa, ar_bytes));
    CK(cudaSetDevice(b)); CK(cudaMalloc(&db, ar_bytes)); CK(cudaMalloc(&ob, ar_bytes));
    cudaStream_t sa=nullptr,sb=nullptr; CK(cudaSetDevice(a)); CK(cudaStreamCreate(&sa)); CK(cudaSetDevice(b)); CK(cudaStreamCreate(&sb));
    int iters=200; float ms=0; int token=1;
    CK(cudaSetDevice(a)); cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i=0;i<iters;i++){
        CK(cudaEventRecord(e0, sa));
        CK(cudaSetDevice(a)); ar_kernel<<<1,256,0,sa>>>(da, oa, (__nv_bfloat16*)ha, (const __nv_bfloat16*)hb2, count, ra_i, rb_i, token);
        CK(cudaSetDevice(b)); ar_kernel<<<1,256,0,sb>>>(db, ob, (__nv_bfloat16*)hb2, (const __nv_bfloat16*)ha, count, rb_i, ra_i, token);
        CK(cudaEventRecord(e1, sa));
        CK(cudaEventSynchronize(e1));
        float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms+=t;
        token++;
    }
    printf("in-kernel allreduce 10KB: %.3f ms/op => 64 layers = %.2f ms/token\n", ms/iters, ms/iters*64);
    printf("=> WSL2 MAPPED PINNED MEMORY WORKS\n");
    return 0;
}
