#!/usr/bin/env bash
grep -n -E "FAILED|error:|Error" /tmp/ninfer_build.log | head -20
echo "=== context ==="
awk '/FAILED/{found=1} found{print; count++} count>40{exit}' /tmp/ninfer_build.log
