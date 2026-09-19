#!/bin/bash
# Poll the service until it answers, then print the per-card footprint.
set -u
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8088/v1/models)
  if [ "$code" = "200" ]; then
    echo "READY used_by_index:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0
  fi
  sleep 2
done
echo "TIMEOUT"
tail -n 6 /home/zhuojun/prof/serve_supervised.log
exit 1
