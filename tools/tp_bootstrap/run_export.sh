#!/bin/bash
cd /tmp
/usr/local/cuda-13.1/bin/nsys export --type sqlite --force-overwrite true --output /tmp/tp2_prof.sqlite /tmp/tp2_prof.nsys-rep 2>&1 | tail -20
ls -la /tmp/tp2_prof* 2>/dev/null