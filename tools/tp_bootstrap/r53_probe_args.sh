#!/bin/bash
cd /home/zhuojun/ninfer/build/tests
for t in ninfer_qwen3_5_tp2_load_test ninfer_qwen3_5_vision_workspace_test ninfer_qwen3_5_visual_scatter_test; do
  echo "=== $t"
  ./$t --help 2>&1 | head -14
done
