#!/bin/bash
set -u
LOG=/home/zhuojun/prof/serve_supervised.log
BEFORE=$(grep -ac '\[mtp\] round' "$LOG" || true)
BEFORE_LINE=$(grep -a '\[mtp\] round' "$LOG" | tail -1 || true)
echo "before rounds=$BEFORE $BEFORE_LINE"
python3 /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/r52_bench.py
AFTER_LINE=$(grep -a '\[mtp\] round' "$LOG" | tail -1 || true)
AFTER=$(grep -ac '\[mtp\] round' "$LOG" || true)
echo "after rounds=$AFTER $AFTER_LINE"
