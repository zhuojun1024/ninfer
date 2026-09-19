#!/bin/bash
set -u
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
SRV=/home/zhuojun/ninfer/build/apps/ninfer-serve
echo "=== CLI rejects an unsupported level ==="
"$SRV" /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --reasoning-effort high 2>&1 | head -2
echo "=== restart with --reasoning-effort medium ==="
bash "$TB/serve_stop.sh"
setsid nohup bash "$TB/serve_supervise.sh" 262144 8088 '--kv-dtype fp8 --temperature 0.7 --top-k 20 --top-p 0.80 --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium' >/dev/null 2>&1 &
for i in $(seq 1 180); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "healthy after ${i}s"; break; }
  sleep 1
done
echo "=== one text request ==="
python3 - <<'PY'
import json, urllib.request
body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "What is 17*23? Reply with the number only."}],
        "max_tokens": 256}
req = urllib.request.Request("http://127.0.0.1:8088/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    data = json.loads(r.read().decode())
usage = data.get("usage", {})
print("completion", usage.get("completion_tokens"),
      "reasoning", (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
print("content", repr((data["choices"][0]["message"].get("content") or "")[:60]))
PY
echo "=== serve lines ==="
grep -aE 'req#[0-9]+ (started|done)' /home/zhuojun/prof/serve_supervised.log | tail -2
echo "=== health ==="
curl -s -m 5 -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8088/health
