#!/bin/sh
# Profile one short TP-2 request with nsys. Uses port 8099 to avoid clashing
# with a serve the user may have running on 8080. POSIX sh (dash-safe).
set -u
cd /home/zhuojun/ninfer
PORT=8099
OUT=/tmp/tp2_prof
NSYS=/usr/local/cuda-13.1/bin/nsys

# Kill any stale instance on the profile port
pkill -f "ninfer-serve.*--port $PORT" 2>/dev/null
sleep 1

# Auto-detect the two sm_120 GPUs using CUDA's own enumeration (serve's
# --devices takes CUDA indices; in WSL2 those differ from nvidia-smi's order).
/usr/local/cuda-13.1/bin/nvcc -O2 -arch=sm_120a \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/cuda_devlist.cu \
  -o /tmp/cuda_devlist -lcuda
/tmp/cuda_devlist > /tmp/cuda_devlist.txt
cat /tmp/cuda_devlist.txt
DEVS=$(awk '$2 == "12.0" {print $1}' /tmp/cuda_devlist.txt | head -2 | paste -sd, -)
if [ -z "$DEVS" ]; then
  echo "no sm_120 GPUs found; aborting"
  exit 1
fi
echo "using CUDA devices: $DEVS"

# Both selected GPUs must be free. Map CUDA index -> nvidia-smi index via UUID.
SMI_IDX=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
for d in $(echo $DEVS | tr ',' ' '); do
  UUID=$(awk -v i=$d '$1 == i {print $3}' /tmp/cuda_devlist.txt)
  SMI=$(echo "$SMI_IDX" | awk -F', ' -v u=$UUID '$2 == u {print $1}')
  if [ -z "$SMI" ]; then
    echo "could not map CUDA device $d to nvidia-smi index; aborting"
    exit 1
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $SMI)
  if [ "$used" -gt 1024 ]; then
    echo "nvidia-smi GPU $SMI (CUDA $d) still has ${used} MiB in use; stop the other serve first"
    exit 1
  fi
done

$NSYS profile -t cuda,nvtx -o $OUT --force-overwrite true -- \
  /home/zhuojun/ninfer/build/apps/ninfer-serve \
  /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices $DEVS --max-context 8192 --port $PORT > /tmp/tp2_prof_serve.log 2>&1 &
NSYS_PID=$!

# Wait for readiness (engine load ~27s + warmup ~4s). Fail fast if the serve
# died during startup (e.g. device guard) instead of polling for 2 minutes.
READY=0
for i in $(seq 1 60); do
  if grep -q 'FATAL' /tmp/tp2_prof_serve.log 2>/dev/null; then
    echo "serve failed during startup:"
    tail -5 /tmp/tp2_prof_serve.log
    kill $NSYS_PID 2>/dev/null
    pkill -f "ninfer-serve.*--port $PORT" 2>/dev/null
    exit 1
  fi
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
    READY=1
    break
  fi
  sleep 2
done
if [ "$READY" != "1" ]; then
  echo "serve did not become ready"
  tail -20 /tmp/tp2_prof_serve.log
  kill $NSYS_PID 2>/dev/null
  exit 1
fi
echo "serve ready, sending profile request"

# One request: short prompt, 24 decode tokens
REQ='{"model":"qwen3.8-27b","messages":[{"role":"user","content":"1+1等于几？只回答数字"}],"max_tokens":24}'
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$REQ" -o /tmp/tp2_prof_resp.json
echo "request done"
head -c 300 /tmp/tp2_prof_resp.json; echo

# Let the request fully drain, then stop nsys (SIGINT -> stats + report)
sleep 2
kill -INT $NSYS_PID
wait $NSYS_PID 2>/dev/null
pkill -f "ninfer-serve.*--port $PORT" 2>/dev/null
echo "profile written to $OUT.nsys-rep"
ls -la $OUT.nsys-rep 2>/dev/null
