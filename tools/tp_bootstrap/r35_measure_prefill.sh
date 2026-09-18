#!/bin/bash
# Measure the live supervised serve: two fresh long prompts, minimal output, report prefill rate.
LOG=/home/zhuojun/prof/serve_supervised.log
URL=http://127.0.0.1:8088/v1/chat/completions
M=qwen3.8-27b
F1=$(printf 'Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. %.0s' $(seq 1 100))
F2=$(printf 'One two three four five six seven eight nine ten eleven twelve thirteen. %.0s' $(seq 1 160))
mk() { printf '{"model":"%s","messages":[{"role":"user","content":"%s Count the words. Answer with one number."}],"max_tokens":8,"stream":false}' "$M" "$1" > "$2"; }
mk "$F1" /tmp/p1.json
mk "$F2" /tmp/p2.json

before=$(grep -c 'req#.*done' "$LOG" 2>/dev/null || echo 0)
echo '--- prompt ~1100 tok ---'
curl -s --max-time 300 -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/p1.json -o /tmp/p1.out -w 'http=%{http_code} total=%{time_total}s\n'
echo '--- prompt ~2300 tok ---'
curl -s --max-time 300 -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/p2.json -o /tmp/p2.out -w 'http=%{http_code} total=%{time_total}s\n'
echo '--- same ~2300 tok prompt again (prefix reuse) ---'
curl -s --max-time 300 -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/p2.json -o /tmp/p2b.out -w 'http=%{http_code} total=%{time_total}s\n'
echo '--- serve log lines ---'
tail -n +$((before + 1)) "$LOG" | grep -E 'req#.*done'
echo '--- throughput lines ---'
grep 'throughput' "$LOG" | tail -2
