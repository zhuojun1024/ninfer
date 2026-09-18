// Print CUDA device indices with name and compute capability, in CUDA's own
// enumeration order. In WSL2 this can differ from nvidia-smi's PCI-based order
// (observed: nvidia-smi [0=5060Ti,1=T10,2=5060Ti] vs CUDA [0=5060Ti,1=5060Ti,2=T10]),
// and serve's --devices takes CUDA indices. The UUID is printed in nvidia-smi's
// 8-4-4-4-12 hex-digit layout (16 bytes) so the caller can map CUDA index ->
// nvidia-smi index for the memory-free check.
#include <cuda.h>
#include <cstdio>

int main() {
    if (cuInit(0) != CUDA_SUCCESS) { return 1; }
    int n = 0;
    if (cuDeviceGetCount(&n) != CUDA_SUCCESS) { return 1; }
    for (int i = 0; i < n; ++i) {
        CUdevice d;
        if (cuDeviceGet(&d, i) != CUDA_SUCCESS) { return 1; }
        char name[256];
        if (cuDeviceGetName(name, sizeof(name), d) != CUDA_SUCCESS) { return 1; }
        int major = 0, minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);
        CUuuid u;
        if (cuDeviceGetUuid(&u, d) != CUDA_SUCCESS) { return 1; }
        const unsigned char* b = reinterpret_cast<const unsigned char*>(u.bytes);
        printf("%d %d.%d GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x %s\n",
               i, major, minor, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
               b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15], name);
    }
    return 0;
}
