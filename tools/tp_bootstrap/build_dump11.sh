#!/bin/bash
set -e
cd /home/zhuojun/ninfer
g++ -std=c++20 -O0 -I src -I include -I third_party \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/dump11.cpp \
  build/src/models/libninfer_model_loading.a \
  build/src/models/libninfer_model_runtime.a \
  build/src/artifact/libninfer_artifact.a \
  build/src/core/libninfer_core.a \
  build/src/text/libninfer_text.a \
  -o /tmp/dump11 -lpthread -lcudart -lcuda
echo "BUILD_OK"
/tmp/dump11 /mnt/d/LLM/qwen3_8_27b_nvfp4.ninfer
