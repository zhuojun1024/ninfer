#!/usr/bin/env bash
export PATH=/usr/local/cuda/bin:$PATH
echo "=== find compiler id files ==="
find /usr/share -name "CMakeCUDACompilerId*" 2>/dev/null
find /usr/lib -name "CMakeCUDACompilerId*" 2>/dev/null
echo "=== which cmake ==="
which cmake
cmake --version | head -1
echo "=== minimal repro: nvcc compiling a cpp with math.h ==="
cat > /tmp/mh.cpp <<'EOF'
#include <math.h>
int main() { return 0; }
EOF
nvcc -x cu /tmp/mh.cpp -o /tmp/mh 2>&1 | head -10
echo "NVCC_EXIT=$?"
echo "=== nvcc verbose host compiler ==="
nvcc -v -x cu /tmp/mh.cpp -o /tmp/mh 2>&1 | grep -E "g\+\+|cicc|host" | head -8
