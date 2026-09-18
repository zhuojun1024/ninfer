#!/bin/bash
# r46: fill the recommended-config row (fp8 KV, 131072 context) in the delivery doc.
set -u
SW=/home/zhuojun/prof
res=$SW/r46_rec.log
: > "$res"
for i in $(seq 1 120); do
  grep -q 'listening on http://127.0.0.1:8088' "$SW/serve_supervised.log" 2>/dev/null && break
  sleep 2
done
grep -a 'capacity |' "$SW/serve_supervised.log" | tail -1 >> "$res"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ' >> "$res"; echo "" >> "$res"
curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @"$SW/r39_prefill_req.json" -o "$SW/r46_prefill.json" \
  -w "prefill http=%{http_code} time_total=%{time_total}\n" >> "$res"
curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @"$SW/r39_decode_req.json" -o "$SW/r46_decode.json" \
  -w "decode http=%{http_code} time_total=%{time_total}\n" >> "$res"
python3 - <<'PY' >> "$res"
import json
for name in ("prefill", "decode"):
    try:
        u = json.load(open("/home/zhuojun/prof/r46_%s.json" % name))["usage"]
        print("%s prompt_tokens=%s completion_tokens=%s" % (name, u.get("prompt_tokens"), u.get("completion_tokens")))
    except Exception as exc:
        print("%s ERROR %s" % (name, exc))
PY
echo R46_DONE >> "$res"
