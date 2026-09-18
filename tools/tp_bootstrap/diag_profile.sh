#!/bin/sh
echo '=== processes ==='
ps aux | grep -E 'nsys|ninfer-serve' | grep -v grep
echo '=== serve log ==='
cat /tmp/tp2_prof_serve.log 2>/dev/null || echo '(no log)'
echo '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
