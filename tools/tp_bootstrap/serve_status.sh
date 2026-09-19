#!/bin/bash
echo "--- ninfer/supervisor processes ---"
pgrep -af 'ninfer' || echo "none"
echo "--- health ---"
curl -s -m 3 -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8088/health 2>/dev/null || echo "no response"
echo "--- gpu memory ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
