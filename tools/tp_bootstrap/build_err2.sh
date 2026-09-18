#!/usr/bin/env bash
ls -la /tmp/ninfer_*.log 2>&1
echo "=== rerun build to capture errors (incremental) ==="
cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda/bin:$PATH
cmake --build build -j 8 2>&1 | grep -B3 -A20 -E "FAILED|error:" | head -100
