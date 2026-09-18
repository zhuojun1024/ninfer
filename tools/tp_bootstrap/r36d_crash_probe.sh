#!/bin/bash
# Crash reproduction: long greedy vs long sampled requests on the MTP route.
set -u
cd /home/zhuojun/prof
: > crash_greedy.json; : > crash_samp.json
for mode in greedy samp; do
  if [ "$mode" = greedy ]; then
    T='"temperature":0,'
  else
    T=''
  fi
  BODY="{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a very long, extremely detailed step-by-step reasoning (at least 1500 words) proving that the sum of the first n odd numbers equals n squared, covering every intermediate detail and every edge case.\"}],$T\"max_tokens\":700}"
  curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$BODY" -o "crash_$mode.json" -w "mode=$mode http=%{http_code} wall=%{time_total}s\n"
  sleep 6
done
for f in crash_greedy crash_samp; do
  if [ -s "$f.json" ]; then
    python3 -c "import json,sys; d=json.load(open('$f.json')); u=d.get('usage',{}); t=d.get('timings',{}); print('$f ok tokens=%s %.1f tok/s finish=%s' % (u.get('completion_tokens'), t.get('predicted_per_second',-1), d['choices'][0].get('finish_reason')))"
  else
    echo "$f EMPTY (server died?)"
  fi
done
echo '--- serve log tail ---'
tail -6 serve_supervised.log
