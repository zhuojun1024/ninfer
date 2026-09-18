#!/bin/bash
cd /home/zhuojun/ninfer
echo '--- how does the serve link ninfer_core? ---'
ldd build/apps/ninfer-serve 2>/dev/null | grep -i ninfer || echo '(no dynamic ninfer lib)'
ls -la build/src/core/libninfer_core.* 2>/dev/null
ls -la build/src/runtime/*/*.so build/src/*/*.so 2>/dev/null | head
echo '--- SASS fingerprint of the AR kernel in the object file ---'
OBJ=build/src/core/CMakeFiles/ninfer_core.dir/tp/device_pair.cu.o
/usr/local/cuda-13.1/bin/cuobjdump -sass $OBJ 2>/dev/null | awk '/ar_inplace_bf16/{f=1} f{print}' | grep -c 'SR_CTAID'
echo '(count of SR_CTAID reads; >0 means the kernel reads blockIdx)'
echo '--- same for the serve binary (may take a moment) ---'
/usr/local/cuda-13.1/bin/cuobjdump -sass build/apps/ninfer-serve 2>/dev/null | awk '/ar_inplace_bf16/{f=1} f{print}' | grep -c 'SR_CTAID'
echo '--- string check: is ar_blocks/slot code present in the binary? ---'
grep -c 'kArMaxBlocks' src/core/tp/device_pair.cu
