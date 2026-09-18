#!/bin/bash
cd /tmp
/usr/local/cuda-13.1/bin/nsys export --type sqlite --force-overwrite true -o /tmp/tp2_prof /tmp/tp2_prof.nsys-rep >/dev/null 2>&1
ls -la /tmp/tp2_prof.sqlite
which python3 || ls /usr/bin/python3*
python3 /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/analyze12.py