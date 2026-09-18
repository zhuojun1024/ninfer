#!/bin/bash
cd /home/zhuojun/ninfer
echo '--- log ---'
ls -l /home/zhuojun/prof/serve_r35_greedy.log
cat /home/zhuojun/prof/serve_r35_greedy.log
echo '--- serve binary ---'
ls -l build/apps/ninfer-serve
echo '--- help grep ---'
./build/apps/ninfer-serve --help 2>&1 | grep -iE 'greedy|temperature|sample' || echo 'no greedy flag in help'
echo '--- processes ---'
pgrep -a -f 'apps/ninfer' || echo none
