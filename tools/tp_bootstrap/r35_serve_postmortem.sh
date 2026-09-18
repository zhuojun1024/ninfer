#!/bin/bash
echo '--- date / uptime ---'
date; uptime
echo '--- ninfer processes ---'
pgrep -a -f 'ninfer' || echo 'NO ninfer process'
echo '--- llama.cpp processes (must be untouched) ---'
pgrep -a -f 'llama' || echo 'no llama process'
echo '--- port 8088 listener ---'
(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -E '8088|State' || echo 'no 8088 listener'
echo '--- serve log tail (final) ---'
tail -30 /home/zhuojun/prof/serve_final.log 2>/dev/null || echo 'no final log'
echo '--- serve log full line count ---'
wc -l /home/zhuojun/prof/serve_final.log 2>/dev/null
echo '--- wsl memory ---'
free -h
echo '--- gpu ---'
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo '--- kernel OOM / kill events ---'
dmesg 2>/dev/null | tail -40 || echo 'dmesg unavailable'
