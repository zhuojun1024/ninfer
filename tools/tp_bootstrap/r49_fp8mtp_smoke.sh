#!/bin/bash
# MTP + fp8 KV smoke test at the recommended sampling settings (a combination never measured).
set -u
cd /home/zhuojun/prof || exit 1
python3 - <<'PY'
import json
body = {"model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "写一篇1000字左右的文章，谈谈你对近代人工智能发展对人类生活的影响。"}],
        "max_tokens": 1000}
open("/home/zhuojun/prof/smoke_req.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
PY
LOG=/home/zhuojun/prof/r49_smoke.log
: > "$LOG"
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer || exit 1
./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 131072 --kv-capacity auto --port 8089 \
  --temperature 0.7 --top-k 20 --top-p 0.80 --spec mtp --draft-tokens 2 >> "$LOG" 2>&1 &
for i in $(seq 1 150); do grep -q "listening on http://127.0.0.1:8089" "$LOG" 2>/dev/null && break; sleep 2; done
for i in 1 2 3; do
  curl -s --max-time 300 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/smoke_req.json -o /home/zhuojun/prof/smoke-$i.json \
    -w "smoke-$i http=%{http_code} wall=%{time_total}s\n" >> "$LOG" 2>&1
done
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
sleep 3
python3 - <<'PY' >> "$LOG" 2>&1
import json, re
for i in (1, 2, 3):
    p = "/home/zhuojun/prof/smoke-%d.json" % i
    try:
        d = json.load(open(p))
    except Exception as exc:
        print("smoke-%d unreadable %s" % (i, exc)); continue
    ch = d["choices"][0]; m = ch["message"]
    c = m.get("content") or ""; r = m.get("reasoning_content") or ""
    zeros = max([len(x) for x in re.findall(r"0+", c)] or [0])
    print("smoke-%d finish=%s content=%d reason=%d longest0=%d" % (
        i, ch.get("finish_reason"), len(c), len(r), zeros))
print("SMOKE_DONE")
PY
