#!/bin/bash
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
out=/home/zhuojun/prof/kvdiag.log
: > "$out"
NINFER_KV_DIAG=1 timeout 150 ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --kv-capacity auto --kv-dtype fp8 --port 8088 >> "$out" 2>&1
echo "exit=$?" >> "$out"
res=/home/zhuojun/prof/kvdiag-out.log
: > "$res"
grep -a 'kvdiag' "$out" | head -3 >> "$res"
grep -aE 'failed:|FATAL' "$out" | tail -3 >> "$res"
echo KVDIAG_DONE >> "$res"
