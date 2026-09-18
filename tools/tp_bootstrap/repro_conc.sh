#!/bin/bash
URL=http://127.0.0.1:8088/v1/chat/completions
M=qwen3.8-27b
P() { printf '{"model":"%s","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"%s"}],"max_tokens":16,"stream":%s}' "$M" "$1" "$2"; }
echo '### 3 concurrent requests (mixed stream) ###'
P 'Say hi in one word.' false > /tmp/p1.json
P 'Name one planet.' true   > /tmp/p2.json
P 'Name one color.' false  > /tmp/p3.json
for i in 1 2 3; do
  ( curl -s -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/p$i.json -o /tmp/r$i.json -w "req$i HTTP=%{http_code} wall=%{time_total}s\n" ) &
done
wait
for i in 1 2 3; do echo "--- r$i ---"; head -c 400 /tmp/r$i.json; echo; done