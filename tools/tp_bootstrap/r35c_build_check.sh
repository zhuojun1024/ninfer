#!/bin/bash
cd /home/zhuojun/ninfer
echo '--- source vs object mtimes ---'
stat -c '%y  %n' src/core/tp/device_pair.cu
find build -name 'device_pair.cu.o' -exec stat -c '%y  %n' {} \; 2>/dev/null
echo '--- ninja dry run (what is dirty) ---'
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 -- -n 2>&1 | head -20
echo '--- sync script ---'
cat /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/build_r35.sh
