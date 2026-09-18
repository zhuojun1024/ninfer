#!/bin/bash
# Statistical A/B: 6 samples per route, temp 0.7, same prompt. Is MTP systematically degenerate?
set -u
cd /home/zhuojun/prof
python3 - <<'PY'
import json
body = {"model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "写一篇1000字左右的文章，谈谈你对近代人工智能发展对人类生活的影响。"}],
        "max_tokens": 1400}
open("/home/zhuojun/prof/essay_req.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
PY
AB=/home/zhuojun/prof/ab_stats.log
: > "$AB"
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
pkill -9 -f ninfer-serve 2>/dev/null || true
sleep 2
cd /home/zhuojun/ninfer
run_route() {
  local tag=$1 port=$2; shift 2
  ./build/apps/ninfer-serve /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --devices 0,1 \
    --max-context 65536 --port "$port" --temperature 0.7 --top-k 20 --top-p 0.80 "$@" >> "$AB" 2>&1 &
  for i in $(seq 1 120); do grep -q "listening on http://127.0.0.1:$port" "$AB" 2>/dev/null && break; sleep 2; done
  for i in $(seq 1 6); do
    local out=/home/zhuojun/prof/st-$tag-$i.json
    curl -s --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
      --data-binary @/home/zhuojun/prof/essay_req.json -o "$out" \
      -w "st-$tag-$i http=%{http_code} wall=%{time_total}s\n" >> "$AB" 2>&1
  done
  pkill -9 -f 'build/apps/ninfer-serve' 2>/dev/null || true
  sleep 3
}
run_route mtp 8088 --spec mtp --draft-tokens 2
run_route plain 8089
python3 - <<'PY' >> "$AB" 2>&1
import json, glob, re, statistics
for tag in ("mtp", "plain"):
    rows = []
    for path in sorted(glob.glob("/home/zhuojun/prof/st-%s-*.json" % tag)):
        try:
            d = json.load(open(path))
        except Exception as exc:
            print(tag, path, "unreadable", exc); continue
        m = d["choices"][0]["message"]
        c = m.get("content") or ""; r = m.get("reasoning_content") or ""
        u = d.get("usage", {})
        zeros = max([len(x) for x in re.findall(r"0+", c)] or [0])
        print("%s tok=%-5s finish=%-8s content=%-5d reason=%-5d longest0=%d" % (
            path.split("/")[-1], u.get("completion_tokens"),
            d["choices"][0].get("finish_reason"), len(c), len(r), zeros))
        rows.append(len(c))
    if rows:
        print("%s SUMMARY n=%d content median=%d min=%d max=%d degenerate(<200)=%d" % (
            tag, len(rows), statistics.median(rows), min(rows), max(rows),
            sum(1 for x in rows if x < 200)))
PY
echo "STATS_DONE" >> "$AB"
