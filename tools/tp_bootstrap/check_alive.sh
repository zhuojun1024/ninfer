#!/bin/bash
echo "now $(date +%H:%M:%S)"
pgrep -a ninfer-serve || echo 'no ninfer-serve process'
ss -ltnp 2>/dev/null | grep 8088 || echo 'port 8088 not listening'
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader