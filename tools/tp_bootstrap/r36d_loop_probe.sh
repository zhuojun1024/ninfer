#!/bin/bash
# Reproduce the repetition loop on the running MTP route with the server's sampling defaults.
set -u
cd /home/zhuojun/prof
BODY='{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a detailed paragraph about water cycle."}],"max_tokens":900}'
curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d "$BODY" -o loop_mtp.json -w "loop_mtp http=%{http_code} wall=%{time_total}s\n"
python3 - <<'PYEOF'
import json, collections
d = json.load(open("loop_mtp.json"))
m = d["choices"][0]["message"]
t = d.get("timings", {}); u = d.get("usage", {})
text = (m.get("reasoning_content") or "") + "||" + (m.get("content") or "")
print("completion=%s predicted_n=%s %.2f tok/s finish=%s" % (
    u.get("completion_tokens"), t.get("predicted_n"), t.get("predicted_per_second", -1),
    d["choices"][0].get("finish_reason")))
print("len(reason)=%d len(content)=%d" % (len(m.get("reasoning_content") or ""), len(m.get("content") or "")))
# repetition: count 3-gram (char-level) and find the most repeated 40-char window
windows = collections.Counter(text[i:i+40] for i in range(0, max(0, len(text)-40), 5))
top = windows.most_common(3)
print("most repeated 40-char windows:", [(w[:40].replace(chr(10), " "), c) for w, c in top])
print("tail 300:", repr(text[-300:]))
PYEOF
