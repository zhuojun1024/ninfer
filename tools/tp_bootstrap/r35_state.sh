#!/bin/bash
echo '--- ninfer processes ---'
pgrep -a -f 'ninfer' || echo none
echo '--- health ---'
curl -s -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8088/health || true
echo '--- greedy log ---'
ls -l /home/zhuojun/prof/serve_r35_greedy.log 2>/dev/null
tail -20 /home/zhuojun/prof/serve_r35_greedy.log 2>/dev/null
echo '--- gpu ---'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo '--- bashes ---'
pgrep -a -f 'r35_greedy' || echo 'no greedy script running'
