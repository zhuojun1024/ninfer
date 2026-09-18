#!/bin/bash
# Profile the batched TP-2 prefill with nsys: one ~1750-token prompt and 4 output tokens, so the
# trace is dominated by prefill chunks. Usage: r35b_profile_prefill.sh [prefill_chunk]
set -u
cd /home/zhuojun/ninfer
PORT=8099
CHUNK="${1:-1024}"
OUT=/home/zhuojun/prof/prefill_prof_c${CHUNK}
NSYS=/usr/local/cuda-13.1/bin/nsys
mkdir -p /home/zhuojun/prof

pkill -9 -f 'ninfer-serve' 2>/dev/null
for _ in $(seq 1 15); do pgrep -f 'ninfer-serve' >/dev/null || break; sleep 1; done
echo '--- gpu before ---'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

/usr/local/cuda-13.1/bin/nvcc -O2 -arch=sm_120a \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/cuda_devlist.cu -o /home/zhuojun/prof/cuda_devlist -lcuda
/home/zhuojun/prof/cuda_devlist > /home/zhuojun/prof/cuda_devlist.txt
DEVS=$(awk '$2 == "12.0" {print $1}' /home/zhuojun/prof/cuda_devlist.txt | head -2 | paste -sd, -)
echo "using CUDA devices: $DEVS chunk=$CHUNK"
[ -n "$DEVS" ] || { echo 'no sm_120 GPUs'; exit 1; }

$NSYS profile -t cuda --sample none -o $OUT --force-overwrite true -- \
  ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices $DEVS --max-context 8192 --port $PORT --prefill-chunk $CHUNK \
  > /home/zhuojun/prof/prefill_prof_serve.log 2>&1 &
NSYS_PID=$!
READY=0
for i in $(seq 1 90); do
  if grep -q 'FATAL' /home/zhuojun/prof/prefill_prof_serve.log 2>/dev/null; then
    echo 'serve failed:'; tail -5 /home/zhuojun/prof/prefill_prof_serve.log
    kill $NSYS_PID 2>/dev/null; pkill -9 -f 'ninfer-serve' 2>/dev/null; exit 1
  fi
  if curl -s --max-time 3 -o /dev/null "http://127.0.0.1:$PORT/health"; then READY=1; break; fi
  sleep 2
done
[ "$READY" = "1" ] || { echo 'serve not ready'; tail -20 /home/zhuojun/prof/prefill_prof_serve.log; kill $NSYS_PID 2>/dev/null; exit 1; }

LONG=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 120))
printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s Count the sentences."}],"max_tokens":4,"stream":false}' "$LONG" > /home/zhuojun/prof/prefill_req.json
echo '--- profile request (fresh long prompt) ---'
curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary @/home/zhuojun/prof/prefill_req.json \
  -o /dev/null -w 'http=%{http_code} total=%{time_total}s\n'
grep -E 'req#1 done' /home/zhuojun/prof/prefill_prof_serve.log
sleep 3
kill -INT $NSYS_PID
wait $NSYS_PID 2>/dev/null
pkill -9 -f 'ninfer-serve' 2>/dev/null
echo "--- report ---"
ls -la $OUT.nsys-rep
$NSYS stats --report cuda_gpu_kern_sum --format csv $OUT.nsys-rep > /home/zhuojun/prof/prefill_kern_sum_c${CHUNK}.csv 2>/dev/null
echo "--- top kernels (by total time) ---"
head -28 /home/zhuojun/prof/prefill_kern_sum_c${CHUNK}.csv
