#!/bin/bash
# Round 53 Step C (retry): rebuild, relaunch the shipped service with --vision, verify text
# neutrality, then send real image requests.
set -u
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
bash "$TB/serve_stop.sh"
echo "=== build ==="
bash "$TB/build_r35.sh" 2>&1 | tail -3
echo "=== launch with --vision ==="
setsid nohup bash "$TB/serve_supervise.sh" 262144 8088 '--kv-dtype fp8 --temperature 0.7 --top-k 20 --top-p 0.80 --spec mtp --draft-tokens 2 --lm-head-draft --vision' >/dev/null 2>&1 &
for i in $(seq 1 180); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "service healthy after ${i}s"; break; }
  sleep 1
done
echo "--- ledger ---"
grep -E '\[mem\]' /home/zhuojun/prof/serve_supervised.log | tail -3
echo "--- fatal tail ---"
tail -3 /home/zhuojun/prof/serve_supervised.log
echo "--- text greedy A/B ---"
bash "$TB/r52_ab.sh" v53b
python3 - <<'PY'
import json
def load(tag):
    return [json.loads(line) for line in open("/home/zhuojun/prof/ab-%s.jsonl" % tag, encoding="utf-8")]
ref = load("embed")
cur = load("v53b")
ok = all(x.get("hash") == y.get("hash") for x, y in zip(ref, cur))
print("v53b", "IDENTICAL to embed" if ok else "DIFFERS")
if not ok:
    for x, y in zip(ref, cur):
        if x.get("hash") != y.get("hash"):
            print("   req%d ref %s(%d) vs %s(%d)" % (x["i"], x["hash"], x["len"], y["hash"], y["len"]))
PY
echo "--- images ---"
python3 "$TB/r53_vision.py" 2>&1 | tail -40
echo "--- idle gpu ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== done ==="
