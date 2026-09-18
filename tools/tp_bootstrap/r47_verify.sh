#!/bin/bash
# r47: run MTP with the Phase::Verify window (env switch) and compare against plain.
set -u
SW=/home/zhuojun/prof
NIN=/home/zhuojun/ninfer
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
res=$SW/r47_verify.log
: > "$res"
log=$SW/r47_mtp.log
: > "$log"
( cd "$NIN" && NINFER_TP2_VERIFY_PHASE=1 ./build/apps/ninfer-serve \
    /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 --max-context 65536 \
    --kv-capacity auto --temperature 0 --spec mtp --draft-tokens 2 --port 8088 ) >> "$log" 2>&1 &
for i in $(seq 1 150); do
  grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break
  pgrep -f 'build/apps/ninfer-serve' >/dev/null || break
  sleep 2
done
if grep -q 'listening on http://127.0.0.1:8088' "$log"; then
  echo "mtp_verify SERVE_OK" >> "$res"
  curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @"$SW/r45_req.json" -o "$SW/r47_mtp.json" \
    -w "mtp_verify http=%{http_code} time_total=%{time_total}\n" >> "$res"
else
  echo "mtp_verify SERVE_FAILED" >> "$res"
  grep -aE 'failed:|FATAL' "$log" | tail -3 >> "$res"
fi
pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
python3 - <<'PY' >> "$res"
import json
def load(p):
    try:
        return json.load(open(p))["choices"][0]["message"].get("content") or ""
    except Exception as exc:
        return "ERR %s" % exc
a = load("/home/zhuojun/prof/r45_plain.json")
b = load("/home/zhuojun/prof/r47_mtp.json")
same = 0
for x, y in zip(a, b):
    if x != y:
        break
    same += 1
print("plain_len=%d mtp_len=%d common_prefix=%d identical=%s" % (len(a), len(b), same, a == b))
PY
echo R47_DONE >> "$res"
