#!/bin/bash
# r44: fp8-KV quality sample against the running 131k fp8 serve (bf16 baseline: 1874/1797/0/0).
set -u
SW=/home/zhuojun/prof
res=$SW/r44_fp8quality.log
: > "$res"
grep -a 'capacity |' "$SW/serve_supervised.log" | tail -1 >> "$res"
for S in 1 2 3 4; do
  curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @"$SW/essay_req.json" -o "$SW/r44_fp8_$S.json" \
    -w "sample=$S http=%{http_code} time_total=%{time_total}\n" >> "$res"
done
python3 - <<'PY' >> "$res"
import glob, json, os, re
for p in sorted(glob.glob("/home/zhuojun/prof/r44_fp8_*.json")):
    try:
        d = json.load(open(p))
        m = d["choices"][0]["message"]
        content = m.get("content") or ""
        runs = [len(x.group(0)) for x in re.finditer(r"0+", content)]
        print("%s content_chars=%d completion=%s longest0=%d" % (
            os.path.basename(p)[:-5], len(content), d.get("usage", {}).get("completion_tokens"),
            max(runs) if runs else 0))
    except Exception as exc:
        print("%s ERROR %s" % (os.path.basename(p), exc))
PY
echo R44_DONE >> "$res"
