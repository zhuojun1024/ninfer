#!/usr/bin/env bash
stat -c '%y %s %n' /home/zhuojun/ninfer/build/apps/ninfer-serve
echo '---WARNINGS---'
grep -nE 'warning:|error:' /tmp/full_build_tp2.log | grep -iE 'tp2_generation|engine\.cpp' | head -20
echo '---DONE---'
