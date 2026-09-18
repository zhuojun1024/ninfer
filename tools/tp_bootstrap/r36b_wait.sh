#!/bin/bash
LOG=/home/zhuojun/prof/serve_supervised.log
for i in $(seq 1 150); do
  if grep -q 'listening on' "$LOG" 2>/dev/null; then echo READY; break; fi
  if grep -qE 'FATAL|terminate called|Aborted' "$LOG" 2>/dev/null; then echo FAILED; break; fi
  sleep 2
done
grep -E 'listening on|FATAL' "$LOG" | tail -2
