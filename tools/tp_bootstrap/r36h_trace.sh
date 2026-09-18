#!/bin/bash
# Greedy token-stream trace on both routes; find the first divergence index.
set -u
cd /home/zhuojun/prof
python3 - <<'PY'
import json
body = {"model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "写一篇1000字左右的文章，谈谈你对近代人工智能发展对人类生活的影响。"}],
        "temperature": 0,
        "max_tokens": 400}
open("/home/zhuojun/prof/trace_req.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
PY
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
run_one() {
  local tag=$1 port=$2; shift 2
  local log=/home/zhuojun/prof/trace-$tag.log
  : > "$log"
  NINFER_TP2_TOKEN_TRACE=1 ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
    --devices 0,1 --max-context 65536 --port "$port" "$@" >> "$log" 2>&1 &
  for i in $(seq 1 120); do grep -q "listening on http://127.0.0.1:$port" "$log" 2>/dev/null && break; sleep 2; done
  curl -s --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/trace_req.json -o "/home/zhuojun/prof/trace-$tag.json" \
    -w "trace-$tag http=%{http_code}\n" >> "$log" 2>&1
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 3
}
run_one plain 8089
run_one mtp 8088 --spec mtp --draft-tokens 2
python3 - <<'PY' >> /home/zhuojun/prof/trace-diff.log 2>&1
import json, re
def toks(path):
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\[toktrace\] (mtp|plain) pos=(\d+) tok=(-?\d+)", line.strip())
        if m:
            out[int(m.group(2))] = int(m.group(3))
    return out
p = toks("/home/zhuojun/prof/trace-plain.log")
m = toks("/home/zhuojun/prof/trace-mtp.log")
print("plain tokens traced:", len(p), " mtp tokens traced:", len(m))
first = None
for i in range(min(len(p), len(m))):
    if p.get(i) != m.get(i):
        first = i
        break
print("first divergence index:", first)
if first is not None:
    lo = max(0, first - 4); hi = first + 5
    print("plain[%d:%d] =" % (lo, hi), [p.get(j) for j in range(lo, hi)])
    print("mtp  [%d:%d] =" % (lo, hi), [m.get(j) for j in range(lo, hi)])
for tag in ("plain", "mtp"):
    try:
        d = json.load(open("/home/zhuojun/prof/trace-%s.json" % tag))
        ch = d["choices"][0]
        msg = ch["message"]
        print("%s finish=%s content=%d reason=%d" % (
            tag, ch.get("finish_reason"), len(msg.get("content") or ""), len(msg.get("reasoning_content") or "")))
    except Exception as exc:
        print(tag, "unreadable", exc)
PY
echo "TRACE_DONE" >> /home/zhuojun/prof/trace-diff.log
