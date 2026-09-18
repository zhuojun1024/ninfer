#!/bin/bash
# Round 35 verification. The TP-2 tests need both cards to themselves, so the engine must already be
# stopped - and this script must never stop it itself: a live supervisor would relaunch it at once and
# the two model loads would collide on GPU memory. Stop It deliberately with serve_stop.sh, run this,
# then relaunch serve_supervise.sh. llama.cpp is never touched.
set -u
if pgrep -f 'ninfer-serve' > /dev/null; then
  echo 'REFUSING: a ninfer-serve process is running.'
  echo 'Run tools/tp_bootstrap/serve_stop.sh first (this also stops its supervisor).'
  pgrep -a -f 'ninfer-serve'
  exit 1
fi
echo '--- remaining ninfer processes ---'
pgrep -a -f 'build/apps/ninfer' || echo none
sleep 2
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo '--- free ---'
free -h | head -2

cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
ART=/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
echo "===== tp2_load_test ====="
./build/tests/ninfer_qwen3_5_tp2_load_test --artifact "$ART"
echo "LOAD_EXIT=$?"
echo "===== tp2_forward_test ====="
./build/tests/ninfer_qwen3_5_tp2_forward_test --artifact "$ART"
echo "FORWARD_EXIT=$?"
echo '--- post-test gpu ---'
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
