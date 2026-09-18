#!/bin/bash
# r43: larger context with quantized KV.
set -u
SW=/home/zhuojun/prof
NIN=/home/zhuojun/ninfer
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
res=$SW/r43_bigctx.log
: > "$res"
try () {
  name=$1; shift
  log=$SW/r43_$name.log
  : > "$log"
  ( cd "$NIN" && ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer "$@" --port 8088 ) >> "$log" 2>&1 &
  for i in $(seq 1 120); do
    grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break
    pgrep -f 'build/apps/ninfer-serve' >/dev/null || break
    sleep 2
  done
  if grep -q 'listening on http://127.0.0.1:8088' "$log"; then
    echo "$name OK" >> "$res"
    grep -a 'capacity |' "$log" | tail -1 >> "$res"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ' >> "$res"; echo "" >> "$res"
    curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
      --data-binary @"$SW/r39_decode_req.json" -o "$SW/r43_$name.json" \
      -w "$name request http=%{http_code} time_total=%{time_total}\n" >> "$res"
  else
    echo "$name FAILED" >> "$res"
    grep -aE 'failed:|FATAL' "$log" | tail -2 >> "$res"
  fi
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 3
}
try fp8_131k --devices 0,1 --max-context 131072 --kv-capacity auto --kv-dtype fp8
try fp8_262k --devices 0,1 --max-context 262144 --kv-capacity auto --kv-dtype fp8
try int8_131k --devices 0,1 --max-context 131072 --kv-capacity auto --kv-dtype int8
echo R43_DONE >> "$res"
