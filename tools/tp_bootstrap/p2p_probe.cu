#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("FAIL %s: %s\n", #x, cudaGetErrorString(e)); return 1;}}while(0)

int main() {
    int n=0; CK(cudaGetDeviceCount(&n));
    printf("device count: %d\n", n);
    // pick the two sm_120 (5060 Ti) devices by CUDA index 0,1
    int a=0, b=1;
    int can_ab=0, can_ba=0;
    cudaDeviceCanAccessPeer(&can_ab, a, b);
    cudaDeviceCanAccessPeer(&can_ba, b, a);
    printf("cudaDeviceCanAccessPeer(%d,%d)=%d  (%d,%d)=%d\n", a,b,can_ab, b,a,can_ba);
    if(!can_ab){ printf("=> no P2P per driver query\n"); return 0; }
    // enable + time a 10KB peer copy both directions
    CK(cudaSetDevice(a)); CK(cudaDeviceEnablePeerAccess(b, 0));
    CK(cudaSetDevice(b)); CK(cudaDeviceEnablePeerAccess(a, 0));
    const size_t bytes = 10240;
    void* pa=nullptr; void* pb=nullptr;
    CK(cudaSetDevice(a)); CK(cudaMalloc(&pa, bytes));
    CK(cudaSetDevice(b)); CK(cudaMalloc(&pb, bytes));
    cudaStream_t sa=nullptr, sb=nullptr;
    CK(cudaSetDevice(a)); CK(cudaStreamCreate(&sa));
    CK(cudaSetDevice(b)); CK(cudaStreamCreate(&sb));
    // warmup
    for(int i=0;i<50;i++){ CK(cudaSetDevice(a)); CK(cudaMemcpyPeerAsync(pa,a,pb,b,bytes,sa)); CK(cudaSetDevice(b)); CK(cudaMemcpyPeerAsync(pb,b,pa,a,bytes,sb)); }
    CK(cudaSetDevice(a)); CK(cudaStreamSynchronize(sa));
    CK(cudaSetDevice(b)); CK(cudaStreamSynchronize(sb));
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int iters=300; float ms=0;
    for(int i=0;i<iters;i++){
        CK(cudaEventRecord(e0, sa));
        CK(cudaSetDevice(a)); CK(cudaMemcpyPeerAsync(pa,a,pb,b,bytes,sa));
        CK(cudaSetDevice(b)); CK(cudaMemcpyPeerAsync(pb,b,pa,a,bytes,sb));
        CK(cudaEventRecord(e1, sa));
        CK(cudaEventSynchronize(e1));
        float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms+=t;
    }
    printf("P2P 10KB round-trip (both dirs): %.3f ms/op  => 64 layers = %.2f ms/token\n", ms/iters, ms/iters*64);
    printf("=> P2P WORKS on WSL2\n");
    return 0;
}
