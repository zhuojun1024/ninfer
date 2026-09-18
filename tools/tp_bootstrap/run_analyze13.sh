#!/bin/bash
set -e
mkdir -p /home/zhuojun/prof
cp /tmp/tp2_prof.nsys-rep /home/zhuojun/prof/
/usr/local/cuda-13.1/bin/nsys export --type sqlite --force-overwrite true --output /home/zhuojun/prof/tp2.sqlite /home/zhuojun/prof/tp2_prof.nsys-rep >/dev/null 2>&1
ls -la /home/zhuojun/prof/
python3 /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/analyze13.py