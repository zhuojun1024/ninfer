#!/bin/bash
# KV dtype sweep: decode quality + speed for bf16 / fp8 / int8 KV, MTP K=3 on.
set -u
SW=/home/zhuojun/prof
NIN=/home/zhuojun/ninfer
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
: > "$SW/kvsweep-out.log"
for DT in bf16 fp8 int8; do
  log="$SW/kvsweep-$DT.log"
  : > "$log"
  ( cd "$NIN" && ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer \
      --devices 0,1 --max-context 65536 --kv-capacity auto --kv-dtype "$DT" --port 8088 \
      --spec mtp --draft-tokens 3 --temperature 0.7 --top-k 20 --top-p 0.80 ) >> "$log" 2>&1 &
  for i in $(seq 1 150); do grep -q 'listening on http://127.0.0.1:8088' "$log" 2>/dev/null && break; sleep 2; done
  if ! grep -q 'listening on http://127.0.0.1:8088' "$log"; then
    echo "dtype=$DT SERVE_FAILED" >> "$SW/kvsweep-out.log"; continue
  fi
  for S in 1 2 3 4; do
    curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions \
      -H 'Content-Type: application/json' --data-binary @"$SW/essay_req.json" \
      -o "$SW/kvsweep-$DT-$S.json" -w "dtype=$DT sample=$S http=%{http_code} time_total=%{time_total}\n" \
      >> "$SW/kvsweep-out.log" 2>&1
  done
  grep -aE 'capacity \|' "$log" | tail -1 >> "$SW/kvsweep-out.log"
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 2
done
python3 - <<'PY' >> "$SW/kvsweep-out.log" 2>&1
import json, glob, re, os
for path in sorted(glob.glob("/home/zhuojun/prof/kvsweep-*-*.json")):
    tag = os.path.basename(path)[:-5]
    try:
        d = json.load(open(path))
        msg = d["choices"][0]["message"]
        content = msg.get("content") or ""
        reason = msg.get("reasoning_content") or ""
        usage = d.get("usage", {})
        runs = [len(m.group(0)) for m in re.finditer(r"0+", content)]
        print("%s content_chars=%d reason_chars=%d completion=%s longest0=%d" % (
            tag, len(content), len(reason), usage.get("completion_tokens"), max(runs) if runs else 0))
    except Exception as exc:
        print("%s ERROR %s" % (tag, exc))
PY
echo KVSWEEP_DONE >> "$SW/kvsweep-out.log"
