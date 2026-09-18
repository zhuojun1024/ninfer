#!/usr/bin/env bash
echo "=== TOOLS ==="
for t in cmake ninja gcc g++ pkg-config make git python3; do
  if command -v $t >/dev/null 2>&1; then
    printf "%-12s %s\n" "$t" "$($t --version 2>/dev/null | head -1)"
  else
    printf "%-12s MISSING\n" "$t"
  fi
done
echo "=== NVCC CANDIDATES ==="
for n in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.8/bin/nvcc /usr/bin/nvcc; do
  [ -x "$n" ] && { echo "$n:"; "$n" --version | tail -1; }
done
echo "=== GPUS IN WSL ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "=== P2P TEST ==="
cat > /tmp/p2p.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("device_count=%d\n", count);
    if (count < 2) { printf("SKIP: fewer than 2 devices\n"); return 0; }
    int ab = 0, ba = 0;
    cudaError_t e1 = cudaDeviceCanAccessPeer(&ab, 0, 1);
    cudaError_t e2 = cudaDeviceCanAccessPeer(&ba, 1, 0);
    printf("can_access_peer(0->1)=%d err=%s\n", ab, cudaGetErrorName(e1));
    printf("can_access_peer(1->0)=%d err=%s\n", ba, cudaGetErrorName(e2));
    if (ab && ba) {
        cudaSetDevice(0);
        cudaError_t e3 = cudaDeviceEnablePeerAccess(1, 0);
        printf("enable_peer(0->1) err=%s\n", cudaGetErrorName(e3));
        // 64 MiB D2D peer copy bandwidth
        size_t bytes = 64ull << 20;
        void *src, *dst;
        cudaSetDevice(1);
        cudaMalloc(&src, bytes);
        cudaMemset(src, 1, bytes);
        cudaSetDevice(0);
        cudaMalloc(&dst, bytes);
        cudaEvent_t s, t;
        cudaEventCreate(&s); cudaEventCreate(&t);
        cudaEventRecord(s);
        cudaMemcpyPeerAsync(dst, 0, src, 1, bytes, 0);
        cudaEventRecord(t);
        cudaEventSynchronize(t);
        float ms = 0;
        cudaEventElapsedTime(&ms, s, t);
        printf("p2d2d_64MiB_ms=%.2f gbps=%.1f\n", ms, (double)bytes / (ms / 1000.0) / 1e9);
        cudaFree(dst);
        cudaSetDevice(1);
        cudaFree(src);
    }
    return 0;
}
EOF
NVCC=$(for n in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.8/bin/nvcc; do [ -x "$n" ] && { echo "$n"; break; }; done)
echo "using nvcc: $NVCC"
"$NVCC" -arch=sm_120a /tmp/p2p.cu -o /tmp/p2p 2>&1 | head -5 && /tmp/p2p
echo "=== SM120A COMPILE CHECK ==="
cat > /tmp/sm120a.cu <<'EOF'
#include <cuda_bf16.h>
__global__ void k(float* x) { x[threadIdx.x] += 1.0f; }
int main() { float* p; cudaMalloc(&p, 4); k<<<1, 1>>>(p); return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1; }
EOF
"$NVCC" -arch=sm_120a /tmp/sm120a.cu -o /tmp/sm120a 2>&1 | head -5 && /tmp/sm120a && echo "sm_120a kernel: OK"
echo "=== PYTHON ==="
python3 --version
