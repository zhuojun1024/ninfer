#!/bin/bash
# r45: does the Phase::Verify window now run, and does MTP then match plain on a greedy request?
set -u
SW=/home/zhuojun/prof
NIN=/home/zhuojun/ninfer
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
res=$SW/r45_verify.log
: > "$res"
run () {
  name=$1; shift
  log=$SW/r45_$name.log
  : > "$log"
  ( cd "$NIN" && ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 \
      --max-context 65536 --kv-capacity auto --temperature 0 "$@" --port 8088 ) >> "$log" 2>&1 &
  for i in $(seq 1 150); do
    grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break
    pgrep -f 'build/apps/ninfer-serve' >/dev/null || break
    sleep 2
  done
  if grep -q 'listening on http://127.0.0.1:8088' "$log"; then
    echo "$name SERVE_OK" >> "$res"
    curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
      --data-binary @"$SW/r45_req.json" -o "$SW/r45_$name.json" \
      -w "$name request http=%{http_code} time_total=%{time_total}\n" >> "$res"
  else
    echo "$name SERVE_FAILED" >> "$res"
    grep -aE 'failed:|FATAL' "$log" | tail -3 >> "$res"
  fi
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 3
}
python3 - <<'PY'
import json
req = json.load(open("/home/zhuojun/prof/essay_req.json"))
req["max_tokens"] = 400
req.pop("temperature", None)
json.dump(req, open("/home/zhuojun/prof/r45_req.json", "w"))
PY
run plain
run mtp --spec mtp --draft-tokens 2
python3 - <<'PY' >> "$res"
import json
try:
    a = json.load(open("/home/zhuojun/prof/r45_plain.json"))["choices"][0]["message"].get("content") or ""
except Exception as e:
    a = "ERR %s" % e
try:
    b = json.load(open("/home/zhuojun/prof/r45_mtp.json"))["choices"][0]["message"].get("content") or ""
except Exception as e:
    b = "ERR %s" % e
same = 0
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        break
    same = i + 1
print("plain_len=%d mtp_len=%d common_prefix=%d identical=%s" % (len(a), len(b), same, a == b))
PY
echo R45_DONE >> "$res"
