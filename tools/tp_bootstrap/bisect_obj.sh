#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
GENCODE="--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]"
INCS="-Iinclude -Isrc"
CUDALIB=/usr/local/cuda-13.1/targets/x86_64-linux/lib
for f in src/core/arena.cu src/core/device.cu src/core/tp/device_pair.cu; do
  base=$(basename "$f" .cu)
  nvcc $GENCODE $INCS -c "$f" -o /tmp/obj_$base.o 2>/dev/null
  g++ /tmp/minlink.cpp /tmp/obj_$base.o -o /tmp/minlink_$base -lcudart -L $CUDALIB 2>/dev/null
  if timeout 20 /tmp/minlink_$base >/dev/null 2>&1; then
    echo "PASS $f"
  else
    echo "CRASH $f"
  fi
done
echo '=== also: all three objects together ==='
g++ /tmp/minlink.cpp /tmp/obj_arena.o /tmp/obj_device.o /tmp/obj_device_pair.o -o /tmp/minlink_all -lcudart -L $CUDALIB 2>/dev/null
timeout 20 /tmp/minlink_all >/dev/null 2>&1 && echo PASS_all || echo CRASH_all