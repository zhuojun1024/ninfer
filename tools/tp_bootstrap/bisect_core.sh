#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
GENCODE="--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]"
INCS="-Iinclude -Isrc"
for f in src/core/*.cu src/core/tp/*.cu; do
  base=$(echo "$f" | sed 's|/|_|g')
  printf '#include "%s"\nint main() { int n = 0; cudaGetDeviceCount(&n); printf("OK loaded\\n"); return 0; }\n' "$f" > /home/zhuojun/ninfer/wrap_$base.cu
  if nvcc $GENCODE $INCS /home/zhuojun/ninfer/wrap_$base.cu -o /tmp/wrap_$base 2>/dev/null; then
    if timeout 20 /tmp/wrap_$base >/dev/null 2>&1; then
      echo "PASS $f"
    else
      echo "CRASH $f"
    fi
  else
    echo "COMPILE_FAIL $f"
  fi
done
rm -f /home/zhuojun/ninfer/wrap_*.cu