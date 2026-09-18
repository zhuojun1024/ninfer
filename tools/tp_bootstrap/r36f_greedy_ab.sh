#!/bin/bash
# Greedy A/B: same prompt, temperature 0 on both routes (removes sampling variance).
set -u
cd /home/zhuojun/prof
python3 - <<'PY'
import json
body = {"model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "写一篇1000字左右的文章，谈谈你对近代人工智能发展对人类生活的影响。"}],
        "temperature": 0,
        "max_tokens": 900}
open("/home/zhuojun/prof/essay_greedy_req.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
PY
AB=/home/zhuojun/prof/ab_greedy.log
: > "$AB"
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 \
  --max-context 65536 --port 8088 --spec mtp --draft-tokens 2 >> "$AB" 2>&1 &
S=$!
for i in $(seq 1 120); do grep -q 'listening on' "$AB" 2>/dev/null && break; sleep 2; done
curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/essay_greedy_req.json -o /home/zhuojun/prof/g_mtp.json \
  -w "g_mtp http=%{http_code} wall=%{time_total}s\n" >> "$AB" 2>&1
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
sleep 2
./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 \
  --max-context 65536 --port 8089 >> "$AB" 2>&1 &
S2=$!
for i in $(seq 1 120); do grep -q 'listening on' "$AB" 2>/dev/null && break; sleep 2; done
curl -s --max-time 300 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/essay_greedy_req.json -o /home/zhuojun/prof/g_plain.json \
  -w "g_plain http=%{http_code} wall=%{time_total}s\n" >> "$AB" 2>&1
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
python3 - <<'PY' >> "$AB" 2>&1
import json, re, collections
for name in ("g_mtp", "g_plain"):
    try:
        d = json.load(open("/home/zhuojun/prof/%s.json" % name))
    except Exception as exc:
        print(name, "unreadable", exc); continue
    m = d["choices"][0]["message"]
    c = m.get("content") or ""; r = m.get("reasoning_content") or ""
    u = d.get("usage", {})
    zeros = max([len(x) for x in re.findall(r"0+", c)] or [0])
    w = collections.Counter(c[i:i+40] for i in range(0, max(0, len(c) - 40), 5))
    top = w.most_common(1)[0] if w else ("", 0)
    print("%s tok=%s finish=%s content=%d reason=%d longest0=%d top40x%d=%r" % (
        name, u.get("completion_tokens"), d["choices"][0].get("finish_reason"),
        len(c), len(r), zeros, top[1], top[0][:24]))
    print("   head:", repr(c[:100]))
PY
echo "AB2_DONE" >> "$AB"
