#!/bin/bash
# Second-domain A/B: Full proposal head vs --lm-head-draft on a code prompt (rare-token risk).
set -u
PROF=/home/zhuojun/prof
MODEL=/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
LOG=$PROF/r50b_code_ab.log
: > "$LOG"
pkill -9 -f 'serve_supervis[e]' 2>/dev/null || true

python3 - <<'PY'
import json
prof = "/home/zhuojun/prof"
req = {"model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "请用 Python 实现一个带过期时间的 LRU 缓存，给出完整代码，并逐段解释实现要点与边界条件。"}],
       "max_tokens": 1400}
json.dump(req, open(prof + "/code_req.json", "w", encoding="utf-8"), ensure_ascii=False)
PY

stop_serve() {
  pkill -9 -f 'build/apps/ninfer-serv[e]' 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f 'build/apps/ninfer-serv[e]' >/dev/null || break; sleep 1; done
}

run_route() {
  local tag="$1"; shift
  local slog="$PROF/r50b_serve_$tag.log"
  : > "$slog"
  stop_serve
  sleep 2
  cd /home/zhuojun/ninfer
  # shellcheck disable=SC2086
  ./build/apps/ninfer-serve "$MODEL" --devices 0,1 \
    --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
    --temperature 0.7 --top-k 20 --top-p 0.80 \
    --spec mtp --draft-tokens 2 --port 8088 $* > "$slog" 2>&1 &
  for _ in $(seq 1 150); do
    grep -q "listening on http://127.0.0.1:8088" "$slog" 2>/dev/null && break
    sleep 2
  done
  if ! grep -q "listening on http://127.0.0.1:8088" "$slog"; then
    echo "=== route $tag FAILED to listen (extra='$*') ===" >> "$LOG"
    tail -30 "$slog" >> "$LOG"
    stop_serve
    return 1
  fi
  echo "=== route $tag listening $(date +%H:%M:%S) extra='$*'" >> "$LOG"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOG" 2>&1
  for i in $(seq 1 6); do
    curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
      --data-binary @"$PROF/code_req.json" -o "$PROF/st3-$tag-$i.json" \
      -w "req-$tag-$i http=%{http_code} wall=%{time_total}s\n" >> "$LOG" 2>&1
  done
  stop_serve
  sleep 3
  return 0
}

run_route full
run_route opt --lm-head-draft
stop_serve
echo "ROUTES_DONE" >> "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1
import glob, json, os, re, statistics
def rows(tag):
    out = []
    for path in sorted(glob.glob("/home/zhuojun/prof/st3-%s-*.json" % tag)):
        try:
            d = json.load(open(path))
        except Exception as exc:
            print(tag, os.path.basename(path), "unreadable", exc); continue
        ch = d["choices"][0]; m = ch.get("message", {})
        c = m.get("content") or ""; r = m.get("reasoning_content") or ""
        t = d.get("timings", {}) or {}
        u = d.get("usage", {}) or {}
        out.append(dict(file=os.path.basename(path), comp=u.get("completion_tokens"),
                        finish=ch.get("finish_reason"), content=len(c), reason=len(r),
                        tok_s=t.get("predicted_per_second", 0.0)))
    return out
for tag in ("full", "opt"):
    rr = rows(tag)
    print("== route %s n=%d" % (tag, len(rr)))
    for x in rr:
        print("   %-14s comp=%-5s finish=%-7s content=%-5d reason=%-5d tok_s=%6.2f" % (
            x["file"], x["comp"], x["finish"], x["content"], x["reason"], x["tok_s"]))
    if rr:
        print("   SUMMARY %s tok_s median=%.2f min=%.2f max=%.2f | content median=%.0f" % (
            tag, statistics.median(x["tok_s"] for x in rr), min(x["tok_s"] for x in rr),
            max(x["tok_s"] for x in rr), statistics.median(x["content"] for x in rr)))
PY
echo "STATS_DONE" >> "$LOG"
