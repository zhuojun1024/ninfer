#!/bin/bash
# Controlled greedy battery for one MTP draft count: 96-token oracle request twice (determinism,
# text comparison against the plain baseline) and a 256-token request twice (decode speed).
set -u
K="${1:?K}"
B96='{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a detailed paragraph about water cycle."}],"max_tokens":96,"temperature":0}'
B256='{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a detailed paragraph about water cycle."}],"max_tokens":256,"temperature":0}'
cd /home/zhuojun/prof
for tag in a b; do
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$B96" -o "mtp_k${K}_${tag}.json" -w "k${K}_96_${tag} http=%{http_code} wall=%{time_total}s\n"
done
for tag in a b; do
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$B256" -o "long_k${K}_${tag}.json" -w "k${K}_256_${tag} http=%{http_code} wall=%{time_total}s\n"
done
python3 - "$K" <<'PYEOF'
import json, sys
k = sys.argv[1]
def load(p):
    d = json.load(open(p))
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + "|" + (m.get("reasoning_content") or ""), d
def firstdiff(x, y):
    for i in range(min(len(x), len(y))):
        if x[i] != y[i]: return i
    return min(len(x), len(y))
plain, _ = load("det_plain_1.json")
a, _ = load("mtp_k%s_a.json" % k)
b, _ = load("mtp_k%s_b.json" % k)
print("K=%s deterministic(96): %s | first diff vs plain: %d (len %d)" % (
    k, a == b, firstdiff(a, plain), min(len(a), len(plain))))
for tag in ("a", "b"):
    _, d = load("long_k%s_%s.json" % (k, tag))
    t = d.get("timings", {}); u = d.get("usage", {})
    n = u.get("completion_tokens")
    print("K=%s long_%s: completion=%s prompt_ms=%.1f predicted_n=%s predicted_ms=%.1f -> %.2f tok/s (wall-based %.2f)" % (
        k, tag, n, t.get("prompt_ms", -1), t.get("predicted_n"), t.get("predicted_ms", -1),
        t.get("predicted_per_second", -1), 1000.0 * n / max(t.get("predicted_ms", 1), 1)))
PYEOF
