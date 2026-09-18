#!/bin/bash
# Profile one short TP-2 request with nsys. Stops any ninfer-serve on 8088/8099 first.
set -u
cd /home/zhuojun/ninfer
PORT=8099
OUT=/tmp/tp2_prof
NSYS=/usr/local/cuda-13.1/bin/nsys

pkill -f 'ninfer-serve' 2>/dev/null
sleep 2
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

/usr/local/cuda-13.1/bin/nvcc -O2 -arch=sm_120a \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/cuda_devlist.cu \
  -o /tmp/cuda_devlist -lcuda
/tmp/cuda_devlist > /tmp/cuda_devlist.txt
cat /tmp/cuda_devlist.txt
DEVS=$(awk '$2 == "12.0" {print $1}' /tmp/cuda_devlist.txt | head -2 | paste -sd, -)
echo "using CUDA devices: $DEVS"

$NSYS profile -t cuda,nvtx -o $OUT --force-overwrite true -- \
  /home/zhuojun/ninfer/build/apps/ninfer-serve \
  /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices $DEVS --max-context 8192 --port $PORT > /tmp/tp2_prof_serve.log 2>&1 &
NSYS_PID=$!

READY=0
for i in $(seq 1 60); do
  if grep -q 'FATAL' /tmp/tp2_prof_serve.log 2>/dev/null; then
    echo 'serve failed during startup:'; tail -5 /tmp/tp2_prof_serve.log
    kill $NSYS_PID 2>/dev/null; pkill -f 'ninfer-serve' 2>/dev/null; exit 1
  fi
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then READY=1; break; fi
  sleep 2
done
if [ "$READY" != "1" ]; then echo 'serve did not become ready'; tail -20 /tmp/tp2_prof_serve.log; kill $NSYS_PID 2>/dev/null; exit 1; fi
sleep 3
echo 'serve ready, sending profile request'
REQ='{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a short paragraph about GPUs."}],"max_tokens":24,"stream":false}'
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$REQ" -o /tmp/tp2_prof_resp.json -w 'HTTP=%{http_code} wall=%{time_total}\n'
grep -o '"timings":{[^}]*}' /tmp/tp2_prof_resp.json; echo
sleep 3
kill -INT $NSYS_PID
wait $NSYS_PID 2>/dev/null
pkill -f 'ninfer-serve' 2>/dev/null
echo "profile written to $OUT.nsys-rep"
ls -la $OUT.nsys-rep 2>/dev/null