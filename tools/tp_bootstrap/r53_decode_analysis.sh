#!/bin/bash
set -u
# 1) summarize the sweep from the service log (one segment per restart)
python3 - <<'PY'
import re
lines = open('/home/zhuojun/prof/serve_supervised.log', encoding='utf-8', errors='replace').read().splitlines()
segments, cur = [], []
for line in lines:
    if 'req#1 started' in line and cur:
        segments.append(cur); cur = []
    cur.append(line)
segments.append(cur)
print("%-10s %-8s %-9s %s" % ("case", "decode", "tok/round", "acceptance"))
for seg in segments[-6:]:
    dones = [l for l in seg if 'done | openai-chat' in l]
    if not dones: continue
    m = re.search(r'decode ([\d.]+) tok/s', dones[-1])
    rounds = [l for l in seg if '[mtp] round' in l]
    extra = "no mtp"
    if len(rounds) > 1:
        pos = [int(re.search(r'pos=(\d+)', l).group(1)) for l in rounds]
        rate = re.findall(r'rate=(\d+)/(\d+)', rounds[-1])[-1]
        out = int(re.search(r'output (\d+)', dones[-1]).group(1))
        extra = "%8.2f %d/%d=%.1f%%" % ((pos[-1]-pos[0])/(len(rounds)-1), int(rate[0]), int(rate[1]),
                                        100.0*int(rate[0])/max(1,int(rate[1])))
    print("%-10s %-8s %s" % ("seg", m.group(1) if m else '?', extra))
PY
# 2) clocks during one decode burst
( timeout 30 nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,utilization.gpu --format=csv,noheader -lms 300 > /tmp/clk2.csv 2>/dev/null ) &
SAMPLER=$!
python3 - <<'PY'
import json, urllib.request
body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Explain paged KV cache in detail."}],
        "max_tokens": 220, "temperature": 0.0, "top_k": 1}
req = urllib.request.Request("http://127.0.0.1:8088/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r: r.read()
print("decode burst finished")
PY
wait $SAMPLER
awk -F, '{gsub(/ /,"",$1); gsub(/ MHz/,"",$2); gsub(/ MHz/,"",$3); gsub(/ W/,"",$4); gsub(/ %/,"",$5);
  i=$1+0; if ($2+0>sm[i]) sm[i]=$2+0; if ($3+0>mem[i]) mem[i]=$3+0; if ($4+0>pw[i]) pw[i]=$4+0; if ($5+0>ut[i]) ut[i]=$5+0}
END { for (i=0;i<=2;i++) if (sm[i]>0) printf "gpu%d max sm=%d MHz mem=%d MHz power=%.1f W util=%d%%\n", i, sm[i], mem[i], pw[i], ut[i] }' /tmp/clk2.csv
