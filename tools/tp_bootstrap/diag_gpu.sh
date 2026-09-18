#!/bin/sh
echo '=== GPU list (names) ==='
nvidia-smi -L
echo '=== index / name / compute_cap / mem ==='
nvidia-smi --query-gpu=index,name,compute_cap,memory.used --format=csv,noheader
echo '=== running serve processes ==='
ps aux | grep -E 'ninfer-serve|nsys' | grep -v grep || echo '(none)'
echo '=== port 8080 ==='
curl -s -m 2 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/health || echo 'down'
