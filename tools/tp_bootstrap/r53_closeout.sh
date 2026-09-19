#!/bin/bash
# Round 53 close-out: run the artifact-backed Vision workspace test with the cards free, then leave
# the shipped 262,144 service running with --vision and confirm it serves an image.
set -u
BIN=/home/zhuojun/ninfer/build/tests
ART=/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
bash "$TB/serve_stop.sh"
echo "=== vision workspace (planner against the real artifact) ==="
NINFER_TEST_ARTIFACT="$ART" "$BIN/ninfer_qwen3_5_vision_workspace_test" 2>&1 | tail -6
echo "EXIT=$?"
echo "=== restart shipped service with --vision ==="
setsid nohup bash "$TB/serve_supervise.sh" 262144 8088 '--kv-dtype fp8 --temperature 0.7 --top-k 20 --top-p 0.80 --spec mtp --draft-tokens 2 --lm-head-draft --vision' >/dev/null 2>&1 &
for i in $(seq 1 180); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "service healthy after ${i}s"; break; }
  sleep 1
done
grep -E '\[mem\] vision|\[mem\] shard 1' /home/zhuojun/prof/serve_supervised.log | tail -2
echo "=== one image request ==="
python3 "$TB/r53_vision.py" /mnt/d/Documents/workbench/ninfer/bench/fixtures/ttft/media/load_05.png 2>&1 | tail -6
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== done ==="
