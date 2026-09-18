#!/usr/bin/env bash
set -o pipefail
cp /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/dump10.cpp /home/zhuojun/ninfer/tools/tp_bootstrap/dump10.cpp
cd /home/zhuojun/ninfer || exit 1
g++ -std=c++20 -O0 -I src -I third_party \
  tools/tp_bootstrap/dump10.cpp \
  build/src/artifact/libninfer_artifact.a build/src/core/libninfer_core.a \
  -o /tmp/dump10 -lpthread 2>&1
rc=$?; echo BUILD_EXIT=$rc; [ $rc -ne 0 ] && exit $rc
/tmp/dump10 /mnt/d/LLM/qwen3_8_27b_nvfp4.ninfer 2>&1
