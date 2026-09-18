#!/usr/bin/env bash
echo "=== tp files present? ==="
ls /home/zhuojun/ninfer/src/core/tp/ 2>&1
ls /home/zhuojun/ninfer/tests/test_tp_device_pair.cpp 2>&1
grep -c "tp/device_pair.cu" /home/zhuojun/ninfer/src/core/CMakeLists.txt
grep -c "ninfer_tp_device_pair_test" /home/zhuojun/ninfer/tests/cmake/CoreTests.cmake
