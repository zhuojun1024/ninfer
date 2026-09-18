#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
for i in $(seq 1 60); do
  used0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
  if [ "$used0" -lt 2000 ] && [ "$used2" -lt 2000 ]; then
    echo "GPU clear after ${i}s: gpu0=${used0}MiB gpu2=${used2}MiB"
    break
  fi
  sleep 1
done
echo "final gpu0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)MiB gpu2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)MiB"
cd /home/zhuojun/ninfer/build
timeout 180 ./tests/ninfer_qwen3_5_tp2_forward_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer 2>&1 | tail -25
echo "TEST_EXIT=${PIPESTATUS[0]}"
