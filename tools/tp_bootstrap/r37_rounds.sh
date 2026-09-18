#!/bin/bash
# Round-level detail: window / drafts / verify argmax / licensed step for the first rounds.
set -u
cd /home/zhuojun/prof
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
log=/home/zhuojun/prof/rounds.log
: > "$log"
NINFER_TP2_ROUND_DETAIL=1 NINFER_TP2_TOKEN_TRACE=1 ./build/apps/ninfer-serve \
  /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 --max-context 65536 --port 8088 \
  --spec mtp --draft-tokens 2 >> "$log" 2>&1 &
for i in $(seq 1 120); do grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break; sleep 2; done
curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/trace_req.json -o /home/zhuojun/prof/rounds.json \
  -w "rounds http=%{http_code}\n" >> "$log" 2>&1
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
python3 - <<'PY' >> /home/zhuojun/prof/rounds-out.log 2>&1
raw = open("/home/zhuojun/prof/rounds.log", errors="replace").read().splitlines()
rounds = [l for l in raw if l.startswith("[round]")]
print("round lines:", len(rounds))
for line in rounds[:14]:
    print(line)
print("--- mtp committed (first 18) ---")
for line in [l for l in raw if l.startswith("[toktrace] mtp")][:18]:
    print(line)
print("--- plain committed (first 18, saved trace) ---")
pt = [l for l in open("/home/zhuojun/prof/trace-plain.log", errors="replace")
      if l.startswith("[toktrace] plain")]
for line in pt[:18]:
    print(line)
PY
echo ROUNDS_DONE >> /home/zhuojun/prof/rounds-out.log
