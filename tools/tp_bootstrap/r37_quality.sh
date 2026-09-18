#!/bin/bash
# Quality sweep at the user's sampling settings: 4 samples per draft count K.
set -u
cd /home/zhuojun/prof
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
for K in 1 2 3; do
  log=/home/zhuojun/prof/qual-k$K.log
  : > "$log"
  ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 \
    --max-context 65536 --port 8088 --spec mtp --draft-tokens $K \
    --temperature 0.7 --top-k 20 --top-p 0.80 >> "$log" 2>&1 &
  for i in $(seq 1 120); do grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break; sleep 2; done
  for i in 1 2 3 4; do
    curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
      --data-binary @/home/zhuojun/prof/essay_req.json -o "/home/zhuojun/prof/qual-k$K-$i.json" \
      -w "k$K-$i http=%{http_code}\n" >> "$log" 2>&1
  done
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 3
done
python3 - <<'PY' >> /home/zhuojun/prof/qual.log 2>&1
import json, glob, re, statistics
for k in (1, 2, 3):
    rows = []
    for path in sorted(glob.glob("/home/zhuojun/prof/qual-k%d-*.json" % k)):
        try:
            d = json.load(open(path))
        except Exception as exc:
            print(path, "unreadable", exc); continue
        ch = d["choices"][0]
        msg = ch["message"]
        c = msg.get("content") or ""
        r = msg.get("reasoning_content") or ""
        zeros = max([len(x) for x in re.findall(r"0+", c)] or [0])
        tok = d.get("usage", {}).get("completion_tokens")
        print("K=%d %s tok=%-5s finish=%-8s content=%-5d reason=%-5d longest0=%d" % (
            k, path.split("/")[-1], tok, ch.get("finish_reason"), len(c), len(r), zeros))
        rows.append(len(c))
    if rows:
        print("K=%d SUMMARY n=%d content median=%d min=%d max=%d degenerate(<200)=%d" % (
            k, len(rows), statistics.median(rows), min(rows), max(rows),
            sum(1 for x in rows if x < 200)))
PY
echo "QUAL_DONE" >> /home/zhuojun/prof/qual.log
