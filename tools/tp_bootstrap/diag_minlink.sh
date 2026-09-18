#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
cat > /tmp/minlink.cpp <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    printf("minlink count=%d\n", n);
    return 0;
}
EOF
g++ /tmp/minlink.cpp -o /tmp/minlink -L src/core -lninfer_core -lcudart -L /usr/local/cuda-13.1/targets/x86_64-linux/lib
/tmp/minlink; echo "MINLINK_EXIT=$?"
echo "=== check what device code ninfer_core carries ==="
cuobjdump -sass src/core/libninfer_core.a 2>/dev/null | grep -c "Function :" || true
readelf -l src/core/libninfer_core.a 2>/dev/null | grep -A2 "CUDA" | head -5 || true
