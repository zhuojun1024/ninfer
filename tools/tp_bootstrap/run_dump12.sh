#!/usr/bin/env bash
set -o pipefail
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer
nvcc -O1 -std=c++20 -I src -I include -I third_party \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/dump12.cpp \
  -o /tmp/dump12 \
  build/src/artifact/libninfer_artifact.a build/src/core/libninfer_core.a \
  build/src/text/libninfer_text.a -lpthread -lcudart -lcuda 2>&1 | tail -5
/tmp/dump12 /mnt/d/LLM/qwen3_8_27b_nvfp4.ninfer
