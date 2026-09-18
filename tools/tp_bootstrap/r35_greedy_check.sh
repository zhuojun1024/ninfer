#!/bin/bash
# Round 35 regression: with greedy sampling the batched (fresh) prefill and the prefix-reuse
# prefill of the same 2063-token prompt must produce byte-identical output.
# Note: WSL2 localhost forwarding can leave a stale relay on a dead port, so every probe is bounded.
set -u
cd /home/zhuojun/ninfer
LOG=/home/zhuojun/prof/serve_r35_greedy.log
URL=http://127.0.0.1:8088/v1/chat/completions
HEALTH=http://127.0.0.1:8088/health
M=qwen3.8-27b

pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null
for _ in $(seq 1 30); do pgrep -f 'build/apps/ninfer-serve' >/dev/null || break; sleep 1; done
if pgrep -f 'build/apps/ninfer-serve' >/dev/null; then echo 'REFUSING: serve still alive'; exit 3; fi
free -h | head -2
setsid nohup ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8088 --greedy < /dev/null > "$LOG" 2>&1 &
echo "launched pid=$!"
ready=0
for i in $(seq 1 120); do
  if ! pgrep -f 'build/apps/ninfer-serve' >/dev/null; then echo 'serve died during startup'; break; fi
  if [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "$HEALTH" 2>/dev/null)" = "200" ]; then
    echo "healthy after ${i}s"; ready=1; break
  fi
  sleep 1
done
if [ "$ready" != "1" ]; then echo 'SERVE DID NOT BECOME HEALTHY'; tail -20 "$LOG"; exit 4; fi

LONG=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 200))
OTHER=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 100))$(printf 'A different tail sentence follows here. %.0s' $(seq 1 100))
mk() { printf '{"model":"%s","messages":[{"role":"user","content":"%s Summarize the above in one short sentence."}],"max_tokens":48,"stream":false,"temperature":0}' "$M" "$1" > "$2"; }
mk "$LONG" /tmp/g1.json
mk "$LONG" /tmp/g2.json
mk "$OTHER" /tmp/g3.json

echo '--- greedy request 1 (fresh batched prefill) ---'
curl -s --max-time 300 -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/g1.json -o /tmp/g1.out -w 'http=%{http_code} total=%{time_total}s\n'
echo '--- greedy request 2 (same prompt, prefix reuse) ---'
curl -s --max-time 300 -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/g2.json -o /tmp/g2.out -w 'http=%{http_code} total=%{time_total}s\n'
echo '--- greedy request 3 (different long prompt, control) ---'
curl -s --max-time 300 -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/g3.json -o /tmp/g3.out -w 'http=%{http_code} total=%{time_total}s\n'

python3 - <<'PY'
import json
def body(path):
    d = json.load(open(path, encoding='utf-8'))
    m = d['choices'][0]['message']
    return (m.get('reasoning_content') or '') + '|' + (m.get('content') or '')
a, b, c = body('/tmp/g1.out'), body('/tmp/g2.out'), body('/tmp/g3.out')
print('req1:', a[:200])
print('req2:', b[:200])
print('req3:', c[:200])
print('MATCH req1==req2:', a == b)
print('DIFFER req1!=req3:', a != c)
PY
grep -E 'req#[0-9]+ done' "$LOG"
