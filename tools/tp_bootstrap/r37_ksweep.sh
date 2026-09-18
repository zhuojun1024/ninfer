#!/bin/bash
# K sweep under the token trace: does the parity failure depend on the draft count?
set -u
cd /home/zhuojun/prof
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
for K in 1 2 3; do
  log=/home/zhuojun/prof/trace-k$K.log
  : > "$log"
  NINFER_TP2_TOKEN_TRACE=1 ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
    --devices 0,1 --max-context 65536 --port 8088 --spec mtp --draft-tokens $K >> "$log" 2>&1 &
  for i in $(seq 1 120); do grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break; sleep 2; done
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/trace_req.json -o "/home/zhuojun/prof/trace-k$K.json" \
    -w "k$K http=%{http_code}\n" >> "$log" 2>&1
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 3
done
python3 - <<'PY' >> /home/zhuojun/prof/trace-ksweep.log 2>&1
import json, re
def toks(path):
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\[toktrace\] (mtp|plain) pos=(\d+) tok=(-?\d+)", line.strip())
        if m:
            out[int(m.group(2))] = int(m.group(3))
    return out
p = toks("/home/zhuojun/prof/trace-plain.log")
print("plain tokens traced:", len(p))
for k in (1, 2, 3):
    m = toks("/home/zhuojun/prof/trace-k%d.log" % k)
    first = None
    for i in range(min(len(p), len(m))):
        if p.get(i) != m.get(i):
            first = i
            break
    try:
        d = json.load(open("/home/zhuojun/prof/trace-k%d.json" % k))
        ch = d["choices"][0]
        content = len(ch["message"].get("content") or "")
        reason = len(ch["message"].get("reasoning_content") or "")
    except Exception:
        content = reason = -1
    print("K=%d traced=%d first_divergence=%s content=%d reason=%d" % (k, len(m), first, content, reason))
PY
echo "KSWEEP_DONE" >> /home/zhuojun/prof/trace-ksweep.log
