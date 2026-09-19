#!/bin/bash
# Is the A/B difference caused by the code change or by the request history the engine carries?
#   warm run, warm run again  -> history-dependent and deterministic?
#   cold restart, run again   -> does it return the golden hash?
set -u
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
bash "$TB/r52_ab.sh" v53d >/dev/null 2>&1
bash "$TB/r52_ab.sh" v53e >/dev/null 2>&1
bash "$TB/serve_stop.sh" >/dev/null 2>&1
setsid nohup bash "$TB/serve_supervise.sh" 262144 8088 '--kv-dtype fp8 --temperature 0.7 --top-k 20 --top-p 0.80 --spec mtp --draft-tokens 2 --lm-head-draft --vision' >/dev/null 2>&1 &
for i in $(seq 1 180); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "cold service healthy after ${i}s"; break; }
  sleep 1
done
bash "$TB/r52_ab.sh" v53f >/dev/null 2>&1
python3 - <<'PY'
import json
def load(tag):
    return [json.loads(line) for line in open("/home/zhuojun/prof/ab-%s.jsonl" % tag, encoding="utf-8")]
tags = ["embed", "v53b", "v53c", "v53d", "v53e", "v53f"]
rows = {t: [r.get("hash") for r in load(t)] for t in tags}
for t in tags:
    print("%-6s %s" % (t, " ".join(str(h)[:8] for h in rows[t])))
print("warm repeat identical:", rows["v53d"] == rows["v53e"])
print("cold equals embed    :", rows["v53f"] == rows["embed"])
PY
