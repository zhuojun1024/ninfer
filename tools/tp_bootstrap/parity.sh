#!/bin/bash
echo "=== serve binary ==="
ls -la --time-style=+%m-%d_%H:%M /home/zhuojun/ninfer/build/apps/ninfer-serve
echo "=== source parity (D: vs WSL) ==="
for f in src/models/qwen3_5/load/tp_split_spec.cpp src/ops/linear/linear.cpp src/ops/linear/fp8/fp8_dispatch.cpp src/ops/linear/fp8/shapes/n5120_k3072.cu src/ops/softmax_attention/dense/causal_cache/geometry.cuh src/ops/softmax_attention/dense/causal_cache/small_t.cu src/ops/wrapper/causal_conv1d_silu.cpp; do
  a=$(md5sum "/mnt/d/Documents/workbench/ninfer/$f" | cut -d' ' -f1)
  b=$(md5sum "/home/zhuojun/ninfer/$f" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "OK   $f"; else echo "DIFF $f"; fi
done
