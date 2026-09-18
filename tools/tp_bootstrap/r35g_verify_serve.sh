#!/bin/bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
for i in $(seq 1 60); do
  if grep -q 'listening on' /home/zhuojun/prof/serve_supervised.log 2>/dev/null; then break; fi
  sleep 2
done
grep -E 'listening on|capacity|weights|workspace|prefill' /home/zhuojun/prof/serve_supervised.log | tail -8
echo '--- health ---'
curl -s --max-time 5 -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8088/health
echo '--- smoke request ---'
curl -s --max-time 60 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"1+1=? answer with the digit only"}],"max_tokens":8}' \
  -o /home/zhuojun/prof/smoke.json -w 'http=%{http_code} wall=%{time_total}s\n'
python3 -c "import json;d=json.load(open('/home/zhuojun/prof/smoke.json'));print('text=',repr(d['choices'][0]['message']['content']));print('timings=',d.get('timings'))"
grep -E 'req#1 done' /home/zhuojun/prof/serve_supervised.log | tail -1
