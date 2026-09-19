#!/bin/bash
set -u
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
bash "$TB/build_r35.sh" 2>&1 | tail -3
echo "=== serve option test binary ==="
ls /home/zhuojun/ninfer/build/tests/ | grep -i 'serve' || true
BIN=$(ls /home/zhuojun/ninfer/build/tests/*serve_options* 2>/dev/null | head -1)
echo "running $BIN"
"$BIN" 2>&1 | tail -25
echo "TEST_EXIT=$?"
