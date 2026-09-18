#!/usr/bin/env bash
set -o pipefail
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
cp "$SRC/src/core/tp/weight_splitter.h" "$DST/src/core/tp/weight_splitter.h"
cp "$SRC/src/core/tp/weight_splitter.cpp" "$DST/src/core/tp/weight_splitter.cpp"
cp "$SRC/src/core/tp/tp_materialize.h" "$DST/src/core/tp/tp_materialize.h"
cp "$SRC/src/core/tp/tp_materialize.cpp" "$DST/src/core/tp/tp_materialize.cpp"
cp "$SRC/src/core/CMakeLists.txt" "$DST/src/core/CMakeLists.txt"
cp "$SRC/src/artifact/materializer.h" "$DST/src/artifact/materializer.h"
cd $DST || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 > /tmp/full_build10.log 2>&1
rc=$?
echo FULL_BUILD_EXIT=$rc
[ $rc -ne 0 ] && { grep -B3 -A18 -E 'FAILED|error:' /tmp/full_build10.log | head -120; }
exit $rc