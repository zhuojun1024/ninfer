#!/bin/bash
pkill -f r35_greedy_check.sh 2>/dev/null
pkill -f r35_state.sh 2>/dev/null
sleep 1
echo '--- orphan scripts ---'
pgrep -a -f 'r35_' || echo none
echo '--- listeners on 8088 ---'
(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -E '8088|State' || echo 'no ss/netstat output'
echo '--- proc net tcp 8088 (hex 1F98) ---'
grep -i ':1F98' /proc/net/tcp /proc/net/tcp6 2>/dev/null || echo 'no socket on 8088'
echo '--- curl with timeout ---'
curl -s --max-time 5 -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8088/health; echo "curl_exit=$?"
echo '--- ninfer processes ---'
pgrep -a -f 'apps/ninfer' || echo none
echo '--- gpu ---'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo '--- greedy log ---'
tail -25 /home/zhuojun/prof/serve_r35_greedy.log 2>/dev/null || echo 'no greedy log'
