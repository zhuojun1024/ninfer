#!/bin/bash
curl -s -o /dev/null -w 'health HTTP=%{http_code}\n' http://127.0.0.1:8088/health
curl -s http://127.0.0.1:8088/v1/models | head -c 200; echo
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
pgrep -a ninfer-serve | head -3