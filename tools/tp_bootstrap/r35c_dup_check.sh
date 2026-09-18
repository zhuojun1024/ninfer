#!/bin/bash
cd /home/zhuojun/ninfer
echo '--- archives containing device_pair.o ---'
for a in $(find build -name '*.a' 2>/dev/null); do n=$(ar t $a 2>/dev/null | grep -c 'device_pair'); if [ "$n" != "0" ]; then echo "$a : $(ar t $a | grep device_pair | tr '\n' ' ')"; fi; done
echo '--- AR kernel symbols in the serve binary ---'
nm -C build/apps/ninfer-serve 2>/dev/null | grep -c 'ar_inplace_bf16'
echo '--- AR kernel function headers in the fatbin (count) + SR_CTAID per copy ---'
/usr/local/cuda-13.1/bin/cuobjdump -sass build/apps/ninfer-serve 2>/dev/null > /home/zhuojun/prof/serve_sass.txt
grep -c 'ar_inplace_bf16' /home/zhuojun/prof/serve_sass.txt
awk '/Function : .*ar_inplace_bf16/{if (n) print "copy", n, "ctaid", c; n++; c=0} /SR_CTAID/{c++} END{if (n) print "copy", n, "ctaid", c}' /home/zhuojun/prof/serve_sass.txt
