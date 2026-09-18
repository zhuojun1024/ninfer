#!/bin/bash
# Per-column parity: chunk verify argmax vs sequential T=1 decode from the same state.
set -u
cd /home/zhuojun/prof
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
log=/home/zhuojun/prof/colparity.log
: > "$log"
NINFER_TP2_PARITY=1 NINFER_TP2_TOKEN_TRACE=1 ./build/apps/ninfer-serve \
  /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 --max-context 65536 --port 8088 \
  --spec mtp --draft-tokens 2 >> "$log" 2>&1 &
for i in $(seq 1 150); do grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break; sleep 2; done
curl -s --max-time 900 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/trace_req.json -o /home/zhuojun/prof/colparity.json \
  -w "colparity http=%{http_code}\n" >> "$log" 2>&1
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
python3 - <<'PY' >> /home/zhuojun/prof/colparity-out.log 2>&1
import collections, re
cnt = collections.Counter()
first = {}
for line in open("/home/zhuojun/prof/colparity.log", errors="replace"):
    m = re.match(r"\[colparity\] pos=(\d+) col=(\d+) chunk=(-?\d+) decode=(-?\d+) same=(\d+)",
                 line.strip())
    if m:
        col = int(m.group(2))
        same = m.group(5) == "1"
        cnt[(col, same)] += 1
        if not same and col not in first:
            first[col] = (int(m.group(1)), int(m.group(3)), int(m.group(4)))
for col in range(4):
    print("col=%d same=%d diff=%d first_diff=%s" % (
        col, cnt.get((col, True), 0), cnt.get((col, False), 0), first.get(col)))
PY
echo COLPARITY_DONE >> /home/zhuojun/prof/colparity-out.log
