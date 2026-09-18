#!/usr/bin/env bash
B=/home/zhuojun/ninfer/build
echo "=== device_link objects present ==="
find $B -name "cmake_device_link.o" 2>/dev/null
echo ""
echo "=== which libs have RDC device_link (from build.ninja) ==="
grep -o "libninfer_[a-z_]*.a" $B/build.ninja | sort -u
echo ""
echo "=== serve link: does it include any cmake_device_link.o or .o besides main.cpp.o? ==="
awk '/^build apps\/ninfer-serve:/,/^$/' $B/build.ninja | head -3 | grep -o "[A-Za-z0-9_./]*\.o" | sort -u
echo ""
echo "=== test link: .o files ==="
awk '/^build tests\/ninfer_qwen3_5_tp2_forward_test:/,/^$/' $B/build.ninja | head -3 | grep -o "[A-Za-z0-9_./]*\.o" | sort -u
