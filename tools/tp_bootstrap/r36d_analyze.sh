#!/bin/bash
cd /home/zhuojun/prof
python3 - <<'PYEOF'
import json, collections
d = json.load(open("pelican_san.json"))
m = d["choices"][0]["message"]
r = m.get("reasoning_content") or ""
c = m.get("content") or ""
u = d.get("usage", {}); t = d.get("timings", {})
print("tokens=%s finish=%s reason=%d content=%d" % (u.get("completion_tokens"), d["choices"][0].get("finish_reason"), len(r), len(c)))
text = r + "||" + c
w = collections.Counter(text[i:i+60] for i in range(0, max(0, len(text)-60), 3))
for win, n in w.most_common(3):
    print("  x%d %r" % (n, win.replace(chr(10), " ")[:60]))
# line-level repetition
lines = [l for l in c.split(chr(10)) if l.strip()]
lc = collections.Counter(lines)
print("  top repeated lines:", [(l[:40], n) for l, n in lc.most_common(3)])
print("  content tail:", repr(c[-200:]))
PYEOF
