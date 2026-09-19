#!/bin/bash
# Final Round 52 verification: regressions with the cards free, then the shipped service at the
# full 262,144 ceiling and the greedy golden comparison against the pre-optimization reference.
set -u
BIN=/home/zhuojun/ninfer/build/tests
ART=/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
LOG=/home/zhuojun/prof/r52_final.log
: > "$LOG"
run() { echo "=== $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; echo "EXIT=$?" >> "$LOG"; }
run "$BIN/ninfer_qwen3_5_tp2_load_test" --artifact "$ART"
run "$BIN/ninfer_qwen3_5_tp2_load_test" --artifact "$ART" --spec mtp
run "$BIN/ninfer_qwen3_5_tp2_load_test" --artifact "$ART" --spec mtp --lm-head-draft
run "$BIN/ninfer_qwen3_5_tp2_forward_test" --artifact "$ART"
run "$BIN/ninfer_embedding_test"
run "$BIN/ninfer_linear_tp2_split_fp8_head_test"
run "$BIN/ninfer_linear_tp2_split_grouped_head_test"
run "$BIN/ninfer_linear_tp2_split_nvfp4_test"
run "$BIN/ninfer_tp_device_pair_test"
grep -E "===|passed|OK |FAIL|EXIT=" "$LOG"
echo "--- restarting the shipped service at 262144 ---"
setsid nohup bash $TB/serve_supervise.sh 262144 8088 '--kv-dtype fp8 --temperature 0.7 --top-k 20 --top-p 0.80 --spec mtp --draft-tokens 2 --lm-head-draft' >/dev/null 2>&1 &
for i in $(seq 1 300); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "service healthy after ${i}s"; break; }
  sleep 1
done
bash $TB/r52_ab.sh final
bash $TB/r52_cmp3.sh
