#!/usr/bin/env bash
echo "=== what's missing in WSL copy ==="
ls /home/zhuojun/ninfer/src/
echo "=== recopy with anchored exclude ==="
cd /home/zhuojun/ninfer
tar -C /mnt/d/Documents/workbench/ninfer --exclude=./.git --exclude=./models -cf - src tests tools cmake apps include bench eval docs third_party 2>/dev/null | tar -C /home/zhuojun/ninfer -xf -
echo "=== verify ==="
ls /home/zhuojun/ninfer/src/
ls /home/zhuojun/ninfer/src/models/ | head -3
