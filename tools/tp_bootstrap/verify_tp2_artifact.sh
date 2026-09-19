#!/bin/bash
# TP-2 integration tests that need the real artifact, after stopping the serve.
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer/build || exit 1
ART=/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
for t in tests/ninfer_qwen3_5_tp2_load_test tests/ninfer_qwen3_5_tp2_forward_test; do
  log=/tmp/$(basename "$t").log
  if ./"$t" --artifact "$ART" > "$log" 2>&1; then
    echo "PASS $t"
  else
    echo "FAIL $t (exit $?)"
    tail -12 "$log"
  fi
done
echo "TP2_ARTIFACT_TESTS_DONE"
