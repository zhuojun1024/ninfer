#!/usr/bin/env bash
set -o pipefail
cd /home/zhuojun/ninfer || exit 1
g++ -std=c++20 -O0 -I src -I third_party \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/dump_artifact.cpp \
  build/src/artifact/libninfer_artifact.a build/src/core/libninfer_core.a \
  -o /tmp/dump_artifact -lpthread 2>&1
rc=$?
echo DUMP_BUILD_EXIT=$rc
[ $rc -ne 0 ] && exit $rc
/tmp/dump_artifact /mnt/d/LLM/qwen3_8_27b_nvfp4.ninfer > /tmp/artifact_dump.txt 2>&1
echo DUMP_RUN_EXIT=$?
wc -l /tmp/artifact_dump.txt
head -60 /tmp/artifact_dump.txt