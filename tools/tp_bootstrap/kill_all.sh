#!/bin/bash
echo '--- killing all ninfer processes ---'
pkill -9 -f ninfer-serve
pkill -9 -f 'build/apps/ninfer'
sleep 3
echo '--- remaining ---'
pgrep -a ninfer || echo 'none'
echo '--- wsl memory ---'
free -h
echo '--- gpu ---'
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader