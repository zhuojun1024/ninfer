#!/bin/bash
S=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_client.sh
URL=http://127.0.0.1:8088/v1/chat/completions
M=qwen3.8-27b
echo '### stop-token termination'
curl -s -X POST "$URL" -H 'Content-Type: application/json' -o /tmp/r_stop.json -w 'HTTP=%{http_code} wall=%{time_total}s\n' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? One word.\"}],\"max_tokens\":256,\"stream\":false}"
head -c 900 /tmp/r_stop.json; echo
echo
echo '### long generation (512 tok cap)'
curl -s -X POST "$URL" -H 'Content-Type: application/json' -o /tmp/r_512.json -w 'HTTP=%{http_code} wall=%{time_total}s\n' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed technical essay (about 400 words) explaining why LLM decoding is memory-bandwidth bound.\"}],\"max_tokens\":512,\"stream\":false}"
head -c 700 /tmp/r_512.json; echo
grep -o '"timings":{[^}]*}' /tmp/r_512.json; echo
echo
echo '### long prompt (~2000 tok in, 64 out)'
bash $S long | tail -3