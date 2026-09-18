#!/usr/bin/env bash
set -o pipefail
cp /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/dump9.cpp /home/zhuojun/ninfer/tools/tp_bootstrap/dump9.cpp
cd /home/zhuojun/ninfer || exit 1
g++ -std=c++20 -O0 -I src -I third_party \
  tools/tp_bootstrap/dump9.cpp \
  build/src/artifact/libninfer_artifact.a build/src/core/libninfer_core.a \
  -o /tmp/dump9 -lpthread 2>&1
rc=$?; echo BUILD_EXIT=$rc; [ $rc -ne 0 ] && exit $rc
/tmp/dump9 /mnt/d/LLM/qwen3_8_27b_nvfp4.ninfer 2>&1 | grep -E "fp8|obj(162|678|730|8|258)" | head -40
