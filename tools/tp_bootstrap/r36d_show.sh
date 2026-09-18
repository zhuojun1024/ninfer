#!/bin/bash
cd /home/zhuojun/prof
python3 - <<'PYEOF'
import json
for name in ("ver_pelican_1", "ver_water"):
    d = json.load(open(name + ".json"))
    m = d["choices"][0]["message"]
    c = m.get("content") or ""
    print("=== %s (finish=%s) ===" % (name, d["choices"][0].get("finish_reason")))
    print(repr(c[:600]))
    print("...tail:", repr(c[-200:]))
PYEOF
echo "=== last rounds ==="
grep 'mtp. round' verify.log | tail -4
echo "=== rounds for water req ==="
grep -c 'mtp. round' verify.log
