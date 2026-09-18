#!/bin/bash
# Round 35 end-to-end: launch exactly one serve (64k context) and measure the batched TP-2
# prefill over HTTP. Leaves the serve running for manual testing.
set -u
S=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_run.sh
LOG=/home/zhuojun/prof/serve_r35.log
URL=http://127.0.0.1:8088/v1/chat/completions
HEALTH=http://127.0.0.1:8088/health
M=qwen3.8-27b
mkdir -p /home/zhuojun/prof

echo '--- stop any previous serve ---'
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null
for _ in $(seq 1 30); do pgrep -f 'build/apps/ninfer-serve' >/dev/null || break; sleep 1; done
if pgrep -f 'build/apps/ninfer-serve' >/dev/null; then
  echo 'REFUSING to start: another ninfer-serve is still alive'; pgrep -a -f 'build/apps/ninfer-serve'; exit 3
fi
echo 'llama.cpp must stay untouched:'; pgrep -a -f llama-server | head -3 || true

echo '--- launch ---'
setsid nohup bash "$S" 65536 < /dev/null > "$LOG" 2>&1 &
echo "launcher pid=$!"
for i in $(seq 1 240); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$HEALTH" 2>/dev/null || true)
  if [ "$code" = "200" ]; then echo "healthy after ${i}s"; break; fi
  sleep 1
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' "$HEALTH" 2>/dev/null)" != "200" ]; then
  echo 'SERVE DID NOT BECOME HEALTHY'; tail -30 "$LOG"; exit 4
fi
echo '--- process / memory / gpu ---'
pgrep -a -f 'build/apps/ninfer' | head -3
free -h | head -2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo '--- warmup (short prompt) ---'
curl -s -X POST "$URL" -H 'Content-Type: application/json' -o /tmp/warm.json \
  -w 'http=%{http_code} total=%{time_total}s\n' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":8,\"stream\":false}"
tail -c 300 /tmp/warm.json; echo

LONG=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 200))
printf '{"model":"%s","messages":[{"role":"user","content":"%s Summarize the above in one short sentence."}],"max_tokens":64,"stream":true}' "$M" "$LONG" > /tmp/long1.json
printf '{"model":"%s","messages":[{"role":"user","content":"%s Summarize the above in one short sentence."}],"max_tokens":64,"stream":false}' "$M" "$LONG" > /tmp/long2.json

echo '--- long prompt, fresh (streaming, ttfb = prefill + first token) ---'
curl -s -N -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/long1.json \
  -o /tmp/long1.sse -w 'http=%{http_code} ttfb=%{time_starttransfer}s total=%{time_total}s\n'
echo "content: $(grep -o '"content":"[^"]*"' /tmp/long1.sse | head -3 | tr '\n' ' ')"

echo '--- long prompt, repeated (prefix reuse) ---'
curl -s -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/long2.json \
  -o /tmp/long2.json.out -w 'http=%{http_code} total=%{time_total}s\n'
head -c 500 /tmp/long2.json.out; echo

echo '--- serve log tail ---'
tail -12 "$LOG"
