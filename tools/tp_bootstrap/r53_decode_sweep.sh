#!/bin/bash
set -u
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
LOG=/home/zhuojun/prof/serve_supervised.log
BASE='--kv-dtype fp8 --temperature 0.7 --top-k 20 --top-p 0.80'
run_case () {
  name=$1; shift
  bash "$TB/serve_stop.sh" >/dev/null 2>&1
  setsid nohup bash "$TB/serve_supervise.sh" 262144 8088 "$BASE $* --vision --reasoning-effort medium" >/dev/null 2>&1 &
  for i in $(seq 1 200); do
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
    [ "$code" = "200" ] && break
    sleep 1
  done
  python3 - "$name" <<'PY'
import json, sys, urllib.request
prompt = ("Write a detailed technical explanation of how a paged KV cache works in an LLM "
          "inference server. Be thorough.")
def post(mt):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": 0.0, "top_k": 1, "top_p": 1.0}
    req = urllib.request.Request("http://127.0.0.1:8088/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())
post(16)
data = post(256)
print(sys.argv[1], "completion", data.get("usage", {}).get("completion_tokens"))
PY
  sleep 1
  echo "--- case $name"
  grep -a 'req#[0-9]* done' "$LOG" | tail -1
  grep -a '\[mtp\]' "$LOG" | tail -1
}
run_case plain-no-mtp ""
run_case mtp-d1 "--spec mtp --draft-tokens 1 --lm-head-draft"
run_case mtp-d2 "--spec mtp --draft-tokens 2 --lm-head-draft"
run_case mtp-d3 "--spec mtp --draft-tokens 3 --lm-head-draft"
run_case mtp-d4 "--spec mtp --draft-tokens 4 --lm-head-draft"
echo "=== restore shipped config ==="
bash "$TB/serve_stop.sh" >/dev/null 2>&1
setsid nohup bash "$TB/serve_supervise.sh" 262144 8088 "$BASE --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium" >/dev/null 2>&1 &
for i in $(seq 1 200); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "shipped config healthy after ${i}s"; break; }
  sleep 1
done
