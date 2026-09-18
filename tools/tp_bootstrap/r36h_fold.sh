#!/bin/bash
# Diagnostic: skip the replay fold and re-trace the MTP route against the saved plain trace.
set -u
cd /home/zhuojun/prof
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
LOG=/home/zhuojun/prof/trace-mtp-nofold.log
: > "$LOG"
NINFER_TP2_TOKEN_TRACE=1 NINFER_TP2_SKIP_FOLD=1 ./build/apps/ninfer-serve \
  /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 --max-context 65536 --port 8088 \
  --spec mtp --draft-tokens 2 >> "$LOG" 2>&1 &
for i in $(seq 1 120); do grep -q 'listening on http://127.0.0.1:8088' "$LOG" 2>/dev/null && break; sleep 2; done
curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/trace_req.json -o /home/zhuojun/prof/trace-mtp-nofold.json \
  -w "nofold http=%{http_code}\n" >> "$LOG" 2>&1
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
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
m = toks("/home/zhuojun/prof/trace-mtp-nofold.log")
print("NOFOLD traced plain=%d mtp=%d" % (len(p), len(m)))
first = None
for i in range(min(len(p), len(m))):
    if p.get(i) != m.get(i):
        first = i
        break
print("NOFOLD first divergence index:", first)
if first is not None:
    lo = max(0, first - 2)
    print("  plain :", [p.get(j) for j in range(lo, first + 4)])
    print("  nofold:", [m.get(j) for j in range(lo, first + 4)])
try:
    d = json.load(open("/home/zhuojun/prof/trace-mtp-nofold.json"))
    ch = d["choices"][0]
    msg = ch["message"]
    print("  nofold finish=%s content=%d reason=%d" % (
        ch.get("finish_reason"), len(msg.get("content") or ""), len(msg.get("reasoning_content") or "")))
except Exception as exc:
    print("  unreadable", exc)
PY
echo "NOFOLD_DONE" >> /home/zhuojun/prof/trace-diff.log
