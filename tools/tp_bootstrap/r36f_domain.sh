#!/bin/bash
# Does using the full vocab as the MTP token domain restore content?
set -u
cd /home/zhuojun/prof
AB=/home/zhuojun/prof/ab_domain.log
: > "$AB"
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 \
  --max-context 65536 --port 8088 --spec mtp --draft-tokens 2 \
  --temperature 0.7 --top-k 20 --top-p 0.80 >> "$AB" 2>&1 &
for i in $(seq 1 120); do grep -q 'listening on http://127.0.0.1:8088' "$AB" 2>/dev/null && break; sleep 2; done
for i in 1 2 3; do
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/essay_req.json -o "/home/zhuojun/prof/dom-$i.json" \
    -w "dom-$i http=%{http_code} wall=%{time_total}s\n" >> "$AB" 2>&1
done
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
python3 - <<'PY' >> "$AB" 2>&1
import json, glob, re
for path in sorted(glob.glob("/home/zhuojun/prof/dom-*.json")):
    try:
        d = json.load(open(path))
    except Exception as exc:
        print(path, "unreadable", exc); continue
    m = d["choices"][0]["message"]
    c = m.get("content") or ""; r = m.get("reasoning_content") or ""
    u = d.get("usage", {})
    zeros = max([len(x) for x in re.findall(r"0+", c)] or [0])
    print("%s tok=%-5s finish=%-8s content=%-5d reason=%-5d longest0=%d" % (
        path.split("/")[-1], u.get("completion_tokens"),
        d["choices"][0].get("finish_reason"), len(c), len(r), zeros))
PY
echo "DOMAIN_DONE" >> "$AB"
