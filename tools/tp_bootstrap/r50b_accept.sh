#!/bin/bash
sleep "${1:-0}"
cat /home/zhuojun/prof/r50b_code_ab.log
echo "--- acceptance ---"
for t in full opt; do
  f=/home/zhuojun/prof/r50b_serve_$t.log
  echo "route $t rounds=$(grep -ac '\[mtp\] round' "$f")"
  grep -a '\[mtp\] round' "$f" | head -1
  grep -a '\[mtp\] round' "$f" | tail -1
done
