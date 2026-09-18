#!/bin/bash
# Verify the ids-offset fix: no crash on the previously fatal requests, plus MTP acceptance rate.
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
LOG=/home/zhuojun/prof/verify.log
: > "$LOG"
cd /home/zhuojun/prof
python3 - <<'PYEOF'
import json
open("/home/zhuojun/prof/water_req.json", "w").write(json.dumps(
    {"model": "qwen3.8-27b",
     "messages": [{"role": "user", "content": "Write a detailed paragraph about water cycle."}],
     "max_tokens": 900}))
PYEOF
cd /home/zhuojun/ninfer
./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8088 --spec mtp --draft-tokens 2 \
  --temperature 0.7 --top-k 20 --top-p 0.80 >> "$LOG" 2>&1 &
SERVE=$!
for i in $(seq 1 200); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  kill -0 $SERVE 2>/dev/null || break
  sleep 2
done
echo "=== ready ===" >> "$LOG"
for attempt in 1 2; do
  curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/pelican_none_req.json -o "/home/zhuojun/prof/ver_pelican_$attempt.json" \
    -w "VER pelican$attempt http=%{http_code} wall=%{time_total}s\n" >> "$LOG" 2>&1 || echo "curl fail" >> "$LOG"
  kill -0 $SERVE 2>/dev/null || { echo "=== DIED after pelican$attempt ===" >> "$LOG"; break; }
done
curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/water_req.json -o /home/zhuojun/prof/ver_water.json \
  -w "VER water http=%{http_code} wall=%{time_total}s\n" >> "$LOG" 2>&1 || echo "curl fail" >> "$LOG"
sleep 3
echo "=== VERIFY_DONE alive=$(kill -0 $SERVE 2>/dev/null && echo yes || echo no) ===" >> "$LOG"
python3 - <<'PYEOF' >> "$LOG" 2>&1
import json, glob
for path in sorted(glob.glob("/home/zhuojun/prof/ver_*.json")):
    try:
        d = json.load(open(path))
    except Exception as exc:
        print(path, "unreadable", exc); continue
    m = d["choices"][0]["message"]
    r = m.get("reasoning_content") or ""; c = m.get("content") or ""
    t = d.get("timings", {}); u = d.get("usage", {})
    print("%s tokens=%s finish=%s reason=%d content=%d %.1f tok/s" % (
        path.split("/")[-1], u.get("completion_tokens"), d["choices"][0].get("finish_reason"),
        len(r), len(c), t.get("predicted_per_second", -1)))
PYEOF
wait $SERVE
