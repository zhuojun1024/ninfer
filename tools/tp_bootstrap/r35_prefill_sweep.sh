#!/bin/bash
# Prefill scaling sweep on the live serve: TTFT vs prompt length at a fixed 256-token chunk.
# The slope gives the marginal cost per chunk; the intercept gives the per-request fixed cost.
LOG=/home/zhuojun/prof/serve_supervised.log
URL=http://127.0.0.1:8088/v1/chat/completions
M=qwen3.8-27b
before=$(grep -c 'req#.*done' "$LOG" 2>/dev/null || echo 0)
for reps in 30 50 70 90 110; do
  F=$(printf 'Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. %.0s' $(seq 1 $reps))
  printf '{"model":"%s","messages":[{"role":"user","content":"%s Count the words. Answer with one number."}],"max_tokens":4,"stream":false}' "$M" "$F" > /tmp/sweep.json
  printf 'reps=%s ' "$reps"
  curl -s --max-time 300 -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/sweep.json \
    -o /dev/null -w 'http=%{http_code} total=%{time_total}s\n'
done
echo '--- serve log ---'
tail -n +$((before + 1)) "$LOG" | grep -E 'req#[0-9]+ done'
