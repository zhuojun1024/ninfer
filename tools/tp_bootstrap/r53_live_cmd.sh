#!/bin/bash
PID=$(pgrep -f 'ninfer-serve.*qwen3_8_27b' | head -2 | tr '\n' ' ')
for p in $PID; do
  printf 'pid %s: ' "$p"
  tr '\0' ' ' < "/proc/$p/cmdline"
  echo
done
