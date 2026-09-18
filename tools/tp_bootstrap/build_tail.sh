#!/usr/bin/env bash
wc -l /tmp/ninfer_build.log
echo "=== last 40 lines ==="
tail -40 /tmp/ninfer_build.log
