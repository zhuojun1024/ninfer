#!/bin/bash
URL=http://127.0.0.1:8088/v1/chat/completions
M=qwen3.8-27b
post() {
  local name="$1"; local payload="$2"
  echo "### $name"
  printf '%s' "$payload" > /tmp/p.json
  curl -s -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/p.json \
    -o /tmp/r.json -w 'HTTP=%{http_code} wall=%{time_total}s\n'
  head -c 700 /tmp/r.json; echo; echo
}
one='{"model":"M","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16,"stream":false}'
two='{"model":"M","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Say hi in one word."}],"max_tokens":16,"stream":false}'
twostream='{"model":"M","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Say hi in one word."}],"max_tokens":16,"stream":true}'
post 'A: 1 message non-stream' "${one//M/$M}"
post 'B: 1 message non-stream (repeat)' "${one//M/$M}"
post 'C: 2 messages non-stream' "${two//M/$M}"
post 'D: 2 messages stream' "${twostream//M/$M}"
post 'E: 2 messages non-stream again' "${two//M/$M}"