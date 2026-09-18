#!/usr/bin/env bash
cat > /tmp/p2p2.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("device_count=%d\n", count);
    int ab = 0, ba = 0;
    cudaDeviceCanAccessPeer(&ab, 0, 2);
    cudaDeviceCanAccessPeer(&ba, 2, 0);
    printf("can_access_peer(0->2)=%d\n", ab);
    printf("can_access_peer(2->0)=%d\n", ba);
    if (ab && ba) {
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(2, 0);
        cudaSetDevice(2);
        cudaDeviceEnablePeerAccess(0, 0);
        size_t bytes = 64ull << 20;
        void *src, *dst;
        cudaSetDevice(2);
        cudaMalloc(&src, bytes);
        cudaMemset(src, 1, bytes);
        cudaSetDevice(0);
        cudaMalloc(&dst, bytes);
        cudaEvent_t s, t;
        cudaEventCreate(&s); cudaEventCreate(&t);
        // warmup
        cudaMemcpyPeerAsync(dst, 0, src, 2, bytes, 0);
        cudaEventRecord(s);
        for (int i = 0; i < 5; i++) cudaMemcpyPeerAsync(dst, 0, src, 2, bytes, 0);
        cudaEventRecord(t);
        cudaEventSynchronize(t);
        float ms = 0;
        cudaEventElapsedTime(&ms, s, t);
        ms /= 5;
        printf("p2p_64MiB_ms=%.2f gbps=%.1f\n", ms, (double)bytes / (ms / 1000.0) / 1e9);
        cudaFree(dst);
        cudaSetDevice(2);
        cudaFree(src);
    } else {
        printf("P2P 0<->2 UNAVAILABLE: need host-staging fallback\n");
    }
    return 0;
}
EOF
/usr/local/cuda/bin/nvcc -arch=sm_120a /tmp/p2p2.cu -o /tmp/p2p2 && /tmp/p2p2
