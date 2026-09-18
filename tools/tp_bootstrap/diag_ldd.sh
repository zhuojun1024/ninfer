#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd /home/zhuojun/ninfer || exit 1
CUDALIB=/usr/local/cuda-13.1/targets/x86_64-linux/lib

echo '=== ldd t1 (PASS) ==='
ldd /tmp/t1 | grep -E 'cudart|cuda|ptxjit'
echo '=== ldd minlink_bare (CRASH) ==='
ldd /tmp/minlink_bare | grep -E 'cudart|cuda|ptxjit'
echo '=== ldd fresh (PASS) ==='
ldd /tmp/fresh | grep -E 'cudart|cuda|ptxjit'

echo '=== run t1 5 times ==='
for i in 1 2 3 4 5; do timeout 10 /tmp/t1 >/dev/null 2>&1 && echo "t1 run $i: PASS" || echo "t1 run $i: CRASH"; done
echo '=== run minlink_bare 5 times ==='
for i in 1 2 3 4 5; do timeout 10 /tmp/minlink_bare >/dev/null 2>&1 && echo "bare run $i: PASS" || echo "bare run $i: CRASH"; done

echo '=== /usr/local/cuda symlink target ==='
readlink -f /usr/local/cuda
echo '=== ptxjitcompiler versions ==='
strings /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 2>/dev/null | grep -E '^1[23]\.' | head -5
find /usr/local/cuda-13.1 -name '*ptxjit*' 2>/dev/null
find /usr/lib/wsl/drivers -name '*ptxjit*' 2>/dev/null