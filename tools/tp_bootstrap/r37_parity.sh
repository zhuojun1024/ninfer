#!/bin/bash
# Parity probe: verify column-0 argmax vs a T=1 decode step from the same pre-verify state.
set -u
cd /home/zhuojun/prof
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
for K in 1 2; do
  log=/home/zhuojun/prof/parity-k$K.log
  : > "$log"
  NINFER_TP2_PARITY=1 NINFER_TP2_TOKEN_TRACE=1 ./build/apps/ninfer-serve \
    /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 --max-context 65536 --port 8088 \
    --spec mtp --draft-tokens $K >> "$log" 2>&1 &
  for i in $(seq 1 120); do grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break; sleep 2; done
  curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @/home/zhuojun/prof/trace_req.json -o "/home/zhuojun/prof/parity-k$K.json" \
    -w "parity-k$K http=%{http_code}\n" >> "$log" 2>&1
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 3
done
python3 - <<'PY' >> /home/zhuojun/prof/parity.log 2>&1
import json, re
for k in (1, 2):
    rounds = same = 0
    bad = []
    for line in open("/home/zhuojun/prof/parity-k%d.log" % k, errors="replace"):
        m = re.match(r"\[parity\] pos=(\d+) t1=(-?\d+) tk=(-?\d+) same=(\d+)", line.strip())
        if m:
            rounds += 1
            if m.group(4) == "1":
                same += 1
            elif len(bad) < 8:
                bad.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    toks = sum(1 for line in open("/home/zhuojun/prof/parity-k%d.log" % k, errors="replace")
               if line.startswith("[toktrace] mtp"))
    print("K=%d parity_rounds=%d identical=%d mismatches=%s" % (k, rounds, same, bad))
    print("   mtp committed tokens traced:", toks)
PY
echo "PARITY_DONE" >> /home/zhuojun/prof/parity.log
