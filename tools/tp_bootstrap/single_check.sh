#!/bin/bash
echo '--- ninfer processes ---'
pgrep -a -f 'build/apps/ninfer' || echo none
echo '--- wsl memory ---'; free -h | head -2
echo '--- gpu ---'; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo '--- health ---'; curl -s -o /dev/null -w 'HTTP=%{http_code}\n' http://127.0.0.1:8088/health