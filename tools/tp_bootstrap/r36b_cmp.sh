#!/bin/bash
set -u
cd /home/zhuojun/prof
python3 - <<'PYEOF'
import json, itertools
def load(p):
    d = json.load(open(p))
    m = d["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", d
def firstdiff(x, y):
    for i in range(min(len(x), len(y))):
        if x[i] != y[i]: return i
    return min(len(x), len(y))
names = {"plain": "det_plain_1.json", "k1": "mtp_k1_a.json", "k2": "mtp_k2_a.json", "k3": "mtp_k3_a.json"}
data = {}
for k, p in names.items():
    c, r, d = load(p)
    data[k] = r + c   # generation order: reasoning first, then content
    u = d.get("usage", {})
    print("%-6s total=%3d reasoning=%3d completion=%s" % (k, len(data[k]), len(r), u.get("completion_tokens")))
    print("       reason[:70]=%r" % r[:70])
print()
for a, b in itertools.combinations(names, 2):
    print("  %-6s vs %-6s : first diff at %d (match %d chars)" % (a, b, firstdiff(data[a], data[b]), firstdiff(data[a], data[b])))
PYEOF
