#!/bin/bash
# r39: plain/MTP benchmark matrix + VRAM (greedy, kv-capacity auto).
set -u
SW=/home/zhuojun/prof
NIN=/home/zhuojun/ninfer
BENCH=$SW/r39
mkdir -p "$BENCH"
python3 - <<'PY'
import json
req = json.load(open("/home/zhuojun/prof/essay_req.json"))
req["messages"][0]["content"] = (req["messages"][0]["content"] + "\n\n") * 12
req["max_tokens"] = 8
json.dump(req, open("/home/zhuojun/prof/r39_prefill_req.json", "w"))
req2 = json.load(open("/home/zhuojun/prof/essay_req.json"))
req2["max_tokens"] = 256
json.dump(req2, open("/home/zhuojun/prof/r39_decode_req.json", "w"))
PY
out=$BENCH/bench-out.log
: > "$out"
run_case () {
  name=$1; shift
  log=$BENCH/$name.log; : > "$log"
  ( cd "$NIN" && ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
      --devices 0,1 --port 8088 "$@" ) >> "$log" 2>&1 &
  for i in $(seq 1 120); do grep -q 'listening on http://127.0.0.1:8088' "$log" && break; sleep 2; done
  if ! grep -q 'listening on http://127.0.0.1:8088' "$log"; then
    echo "$name SERVE_FAILED" >> "$out"; tail -2 "$log" >> "$out"
    pkill -9 -f 'build/apps/ninfer-serve'; sleep 2; return
  fi
  grep -a 'capacity |' "$log" | tail -1 >> "$out"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ' >> "$out"
  echo "" >> "$out"
  for w in prefill decode; do
    curl -s --max-time 900 http://127.0.0.1:8088/v1/chat/completions \
      -H 'Content-Type: application/json' --data-binary @"$SW/r39_${w}_req.json" \
      -o "$BENCH/$name-${w}.json" -w "$name $w http=%{http_code} time_total=%{time_total}\n" >> "$out"
  done
  pkill -9 -f 'build/apps/ninfer-serve'; sleep 2
}
run_case plain64 --max-context 65536 --kv-capacity auto --temperature 0
run_case mtp64 --max-context 65536 --kv-capacity auto --temperature 0 --spec mtp --draft-tokens 3
run_case plain131 --max-context 131072 --kv-capacity auto --temperature 0
python3 - <<'PY' >> "$out"
import glob, json, os
for p in sorted(glob.glob("/home/zhuojun/prof/r39/*-*.json")):
    try:
        d = json.load(open(p)); u = d.get("usage", {})
        print("%s completion=%s" % (os.path.basename(p)[:-5], u.get("completion_tokens")))
    except Exception as exc:
        print("%s ERROR %s" % (os.path.basename(p), exc))
PY
echo R39_BENCH_DONE >> "$out"
