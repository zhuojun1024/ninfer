#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
LIB=$(find build -name "libninfer_core.a" | head -1)
echo "lib=$LIB"
g++ /tmp/minlink.cpp -o /tmp/minlink -L "$(dirname $LIB)" -lninfer_core -lcudart -L /usr/local/cuda-13.1/targets/x86_64-linux/lib
/tmp/minlink; echo "MINLINK_EXIT=$?"
echo "=== device sections in the lib ==="
nm "$LIB" 2>/dev/null | grep -i "nv_" | head -5
readelf -S "$LIB" 2>/dev/null | grep -i cuda | head -5
