#!/bin/bash
# r40: scope the quantized-KV failures (single device vs TP-2, fp8/int8/nvfp4).
set -u
SW=/home/zhuojun/prof
NIN=/home/zhuojun/ninfer
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
out=$SW/r40_scope.log
: > "$out"
try () {
  name=$1; shift
  log=$SW/r40_$name.log
  : > "$log"
  ( cd "$NIN" && ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer "$@" --port 8088 ) >> "$log" 2>&1 &
  for i in $(seq 1 120); do
    grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break
    pgrep -f 'build/apps/ninfer-serve' >/dev/null || break
    sleep 2
  done
  if grep -q 'listening on http://127.0.0.1:8088' "$log"; then
    echo "$name OK" >> "$out"
    grep -a 'capacity |' "$log" | tail -1 >> "$out"
  else
    echo "$name FAILED" >> "$out"
    grep -aE 'failed:|FATAL|ERROR' "$log" | tail -3 >> "$out"
  fi
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 3
}
try single_bf16 --devices 0 --max-context 32768 --kv-capacity 32768
try single_fp8 --devices 0 --max-context 32768 --kv-capacity 32768 --kv-dtype fp8
try single_int8 --devices 0 --max-context 32768 --kv-capacity 32768 --kv-dtype int8
try tp2_nvfp4 --devices 0,1 --max-context 65536 --kv-capacity auto --kv-dtype nvfp4
echo R40_DONE >> "$out"
