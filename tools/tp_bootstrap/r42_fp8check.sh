#!/bin/bash
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
res=/home/zhuojun/prof/r42_fp8check.log
: > "$res"
for DT in fp8 nvfp4 int8; do
  log=/home/zhuojun/prof/r42_$DT.log
  : > "$log"
  ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 \
    --max-context 65536 --kv-capacity auto --kv-dtype "$DT" --port 8088 >> "$log" 2>&1 &
  for i in $(seq 1 120); do
    grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break
    pgrep -f 'build/apps/ninfer-serve' >/dev/null || break
    sleep 2
  done
  if grep -q 'listening on http://127.0.0.1:8088' "$log"; then
    echo "$DT OK" >> "$res"
    grep -a 'capacity |' "$log" | tail -1 >> "$res"
    curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
      --data-binary @/home/zhuojun/prof/r39_decode_req.json -o /home/zhuojun/prof/r42_$DT.json \
      -w "$DT request http=%{http_code} time_total=%{time_total}\n" >> "$res"
  else
    echo "$DT FAILED" >> "$res"
    grep -aE 'failed:|FATAL' "$log" | tail -2 >> "$res"
  fi
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 3
done
echo R42_DONE >> "$res"
