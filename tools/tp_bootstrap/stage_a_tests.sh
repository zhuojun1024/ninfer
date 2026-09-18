#!/bin/bash
echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,compute_cap --format=csv,noheader
echo '=== compute apps ==='
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo '=== serve binary ==='
ls -la --time-style=+%m-%d_%H:%M /home/zhuojun/ninfer/build/apps/ninfer-serve
echo '=== test binaries ==='
ls /home/zhuojun/ninfer/build/tests/ | head -50
echo '=== source parity (D vs WSL) ==='
cd /home/zhuojun/ninfer
for f in src/models/qwen3_5/load/tp_split_spec.cpp src/models/qwen3_5/execution/text.cpp src/ops/linear/fp8/fp8_dispatch.cpp src/ops/softmax_attention/dense/causal_cache/geometry.cuh src/ops/launcher/causal_conv1d.cu; do
  a=$(md5sum /home/zhuojun/ninfer/$f | cut -d' ' -f1)
  b=$(md5sum /mnt/d/Documents/workbench/ninfer/$f | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "OK   $f"; else echo "DIFF $f"; fi
done
export PATH=/usr/local/cuda-13.1/bin:$PATH
echo '=== load test ==='
./build/tests/ninfer_qwen3_5_tp2_load_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer > /tmp/stage_a_load.log 2>&1
echo "LOAD_EXIT=$?"
tail -4 /tmp/stage_a_load.log
echo '=== forward test ==='
./build/tests/ninfer_qwen3_5_tp2_forward_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer > /tmp/stage_a_fwd.log 2>&1
echo "FWD_EXIT=$?"
tail -4 /tmp/stage_a_fwd.log