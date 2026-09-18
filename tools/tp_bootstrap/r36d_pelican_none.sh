#!/bin/bash
# Same pelican request with reasoning effort disabled, on the plain route.
set -u
cd /home/zhuojun/prof
python3 - <<'PYEOF'
import json
body = {"model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "创建一个HTML，内容是SVG绘制一个鹈鹕骑自行车的2D动画"}],
        "reasoning_effort": "none",
        "max_tokens": 1200}
open("/home/zhuojun/prof/pelican_none_req.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
PYEOF
timeout 600 curl -s --max-time 550 http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/pelican_none_req.json -o /home/zhuojun/prof/pelican_none.json \
  -w 'pelican_none http=%{http_code} wall=%{time_total}s\n' || echo 'curl timeout/failed'
python3 - <<'PYEOF'
import json
try:
    d = json.load(open("pelican_none.json"))
except Exception as exc:
    print("no response:", exc); raise SystemExit(0)
m = d["choices"][0]["message"]
r = m.get("reasoning_content") or ""; c = m.get("content") or ""
u = d.get("usage", {}); t = d.get("timings", {})
print("tokens=%s finish=%s reason=%d content=%d %.1f tok/s" % (
    u.get("completion_tokens"), d["choices"][0].get("finish_reason"), len(r), len(c),
    t.get("predicted_per_second", -1)))
print("content head:", repr(c[:200]))
print("content tail:", repr(c[-200:]))
PYEOF
