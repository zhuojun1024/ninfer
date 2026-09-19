#!/bin/bash
# Same experiment with a long TEXT-only preceding request: if the warm A/B shifts the same way, the
# trigger is the history-derived rewind depth (pre-existing), not the Vision/MTP change.
set -u
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
bash "$TB/serve_stop.sh"
setsid nohup bash "$TB/serve_supervise.sh" 262144 8088 '--kv-dtype fp8 --temperature 0.7 --top-k 20 --top-p 0.80 --spec mtp --draft-tokens 2 --lm-head-draft --vision' >/dev/null 2>&1 &
for i in $(seq 1 180); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "healthy after ${i}s"; break; }
  sleep 1
done
python3 - <<'PY'
import json, urllib.request
body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content":
        "Repeat the following line once: " + ("the quick brown fox jumps over the lazy dog. " * 90)}],
        "max_tokens": 8, "temperature": 0.0}
request = urllib.request.Request("http://127.0.0.1:8088/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=900) as response:
    data = json.loads(response.read().decode())
print("long text request prompt_tokens", data.get("usage", {}).get("prompt_tokens"))
PY
bash "$TB/r52_ab.sh" v53g >/dev/null 2>&1
python3 - <<'PY'
import json
def load(tag):
    return [json.loads(line) for line in open("/home/zhuojun/prof/ab-%s.jsonl" % tag, encoding="utf-8")]
for tag in ("embed", "v53c", "v53g"):
    print("%-6s %s" % (tag, " ".join(str(r.get("hash"))[:8] for r in load(tag))))
print("text history shifts req0 the same way:", load("v53g")[0]["hash"] == load("v53c")[0]["hash"])
PY
