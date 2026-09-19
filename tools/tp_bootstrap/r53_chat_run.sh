#!/bin/bash
set -u
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
bash "$TB/serve_stop.sh"
echo "=== build ==="
bash "$TB/build_r35.sh" 2>&1 | tail -3
echo "=== launch ==="
setsid nohup bash "$TB/serve_supervise.sh" 262144 8088 '--kv-dtype fp8 --temperature 0.7 --top-k 20 --top-p 0.80 --spec mtp --draft-tokens 2 --lm-head-draft --vision' >/dev/null 2>&1 &
for i in $(seq 1 180); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "healthy after ${i}s"; break; }
  sleep 1
done
echo "=== image turn then follow-up turn ==="
python3 "$TB/r53_chat.py" 2>&1 | tail -12
echo "=== serve request lines ==="
grep -aE 'req#[0-9]+ done' /home/zhuojun/prof/serve_supervised.log | tail -3
echo "=== mtp rounds ==="
grep -a '\[mtp\]' /home/zhuojun/prof/serve_supervised.log | tail -2
echo "=== done ==="
