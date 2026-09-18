#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
GENCODE="--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]"
INCS="-Iinclude -Isrc"
printf '#include "src/core/arena.cu"\nint main() { return 0; }\n' > wrap_arena.cu
echo '=== arena.cu ==='
nvcc $GENCODE $INCS wrap_arena.cu -o /tmp/wrap_arena 2>&1 | head -15
printf '#include "src/core/tp/device_pair.cu"\nint main() { return 0; }\n' > wrap_dp.cu
echo '=== tp/device_pair.cu ==='
nvcc $GENCODE $INCS wrap_dp.cu -o /tmp/wrap_dp 2>&1 | head -15
rm -f wrap_arena.cu wrap_dp.cu