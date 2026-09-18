#!/bin/bash
PORT=8088
URL=http://127.0.0.1:$PORT/v1/chat/completions
M=qwen3.8-27b
which="${1:-short}"

post() {
  local name="$1"
  local payload="$2"
  echo "### $name"
  printf '%s' "$payload" > /tmp/payload.json
  curl -s -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/payload.json \
    -o /tmp/resp.json -w 'HTTP=%{http_code} wall=%{time_total}s\n'
  head -c 1800 /tmp/resp.json; echo
}

case "$which" in
  short)
    post 'short (32 tok)' "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=? Answer with just the number.\"}],\"max_tokens\":32,\"stream\":false}"
    ;;
  mid)
    post 'mid (128 tok)' "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a haiku about tensor parallelism.\"}],\"max_tokens\":128,\"stream\":false}"
    ;;
  stop)
    post 'stop-token termination' "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? One word.\"}],\"max_tokens\":256,\"stream\":false}"
    ;;
  long)
    LONG=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 200))
    long=$(printf '{"model":"%s","messages":[{"role":"user","content":"%s Summarize the above in one short sentence."}],"max_tokens":64,"stream":false}' "$M" "$LONG")
    post 'long-prompt (~2000 tok, 64 out)' "$long"
    ;;
  *) echo "unknown case $which"; exit 2;;
esac