#!/bin/bash
cd /home/zhuojun/prof
python3 - <<'PYEOF'
import json
for name in ["det_plain_1", "mtp_k1_a", "mtp_k2_a", "mtp_k3_a", "long_k1_a", "long_k2_a", "long_k3_a"]:
    d = json.load(open(name + ".json"))
    t = d.get("timings", {}); u = d.get("usage", {})
    print("%-11s completion=%3s cache_n=%3s prompt_n=%3s predicted_n=%3s predicted_ms=%8.1f -> %5.2f tok/s" % (
        name, u.get("completion_tokens"), t.get("cache_n"), t.get("prompt_n"),
        t.get("predicted_n"), t.get("predicted_ms", -1), t.get("predicted_per_second", -1)))
PYEOF
