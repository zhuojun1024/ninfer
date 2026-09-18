#!/bin/bash
# A/B: same essay prompt, MTP route vs plain route, same sampling params. Look for token-0 loops.
set -u
cd /home/zhuojun/prof
python3 - <<'PY'
import json
body = {"model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "写一篇1000字左右的文章，谈谈你对近代人工智能发展对人类生活的影响。"}],
        "max_tokens": 1400}
open("/home/zhuojun/prof/essay_req.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
PY
AB=/home/zhuojun/prof/ab.log
: > "$AB"
for i in 1 2; do
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/essay_req.json -o "/home/zhuojun/prof/mtp_essay_$i.json" \
    -w "mtp$i http=%{http_code} wall=%{time_total}s\n" >> "$AB" 2>&1
done
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
LOG=/home/zhuojun/prof/plain_ab.log
: > "$LOG"
cd /home/zhuojun/ninfer
./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --max-context 65536 --port 8089 \
  --temperature 0.7 --top-k 20 --top-p 0.80 >> "$LOG" 2>&1 &
S=$!
for i in $(seq 1 120); do
  grep -q 'listening on' "$LOG" 2>/dev/null && break
  sleep 2
done
for i in 1 2; do
  curl -s --max-time 300 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/essay_req.json -o "/home/zhuojun/prof/plain_essay_$i.json" \
    -w "plain$i http=%{http_code} wall=%{time_total}s\n" >> "$AB" 2>&1
done
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
python3 - <<'PY' >> "$AB" 2>&1
import json, glob, re, collections
for path in sorted(glob.glob("/home/zhuojun/prof/*_essay_*.json")):
    try:
        d = json.load(open(path))
    except Exception as exc:
        print(path, "unreadable", exc); continue
    m = d["choices"][0]["message"]
    c = m.get("content") or ""; r = m.get("reasoning_content") or ""
    u = d.get("usage", {}); t = d.get("timings", {})
    zeros = max([len(x) for x in re.findall(r"0+", c)] or [0])
    w = collections.Counter(c[i:i+40] for i in range(0, max(0, len(c) - 40), 5))
    top = w.most_common(1)[0] if w else ("", 0)
    print("%s tok=%s finish=%s content=%d reason=%d longest0run=%d top40x%d=%r %.1ftps" % (
        path.split("/")[-1], u.get("completion_tokens"), d["choices"][0].get("finish_reason"),
        len(c), len(r), zeros, top[1], top[0][:24], t.get("predicted_per_second", -1)))
PY
echo "AB_DONE" >> "$AB"
