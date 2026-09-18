#!/usr/bin/env bash
LOG=/home/zhuojun/ninfer/build/CMakeFiles/CMakeConfigureLog.yaml
echo "=== log exists ==="
ls -la "$LOG" 2>/dev/null
echo "=== failing command context ==="
grep -n "iseqsig" "$LOG" | head -3
N=$(grep -n "iseqsig" "$LOG" | head -1 | cut -d: -f1)
sed -n "$((N-40)),$((N+5))p" "$LOG"
