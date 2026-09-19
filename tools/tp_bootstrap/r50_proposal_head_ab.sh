#!/bin/bash
# Route A/B: Full proposal head vs --lm-head-draft (optimized indexed head).
# Fixed TP-2 config: 2x RTX 5060 Ti, fp8 KV, 131072 context, MTP K=2, temp 0.7/top-k 20/top-p 0.80.
set -u
PROF=/home/zhuojun/prof
MODEL=/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
LOG=$PROF/r50_proposal_ab.log
: > "$LOG"
# Bracket pattern keeps pkill from matching its own command line.
pkill -9 -f 'serve_supervis[e]' 2>/dev/null || true

python3 - <<'PY'
import json
prof = "/home/zhuojun/prof"
essay = {"model": "qwen3.8-27b",
         "messages": [{"role": "user", "content": "写一篇1000字左右的文章，谈谈你对近代人工智能发展对人类生活的影响。"}],
         "max_tokens": 1400}
json.dump(essay, open(prof + "/essay_req.json", "w", encoding="utf-8"), ensure_ascii=False)
live = {"model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "用一句话说明什么是张量并行。"}],
        "max_tokens": 64}
json.dump(live, open(prof + "/live_req.json", "w", encoding="utf-8"), ensure_ascii=False)
PY

stop_serve() {
  pkill -9 -f 'build/apps/ninfer-serv[e]' 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f 'build/apps/ninfer-serv[e]' >/dev/null || break; sleep 1; done
}

run_route() {
  local tag="$1"; shift
  local slog="$PROF/r50_serve_$tag.log"
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
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >> "$LOG" 2>&1
  curl -s --max-time 300 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary @"$PROF/live_req.json" -o "$PROF/r50_warm_$tag.json" \
    -w "warm-$tag http=%{http_code} wall=%{time_total}s\n" >> "$LOG" 2>&1
  for i in $(seq 1 6); do
    curl -s --max-time 600 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
      --data-binary @"$PROF/essay_req.json" -o "$PROF/st2-$tag-$i.json" \
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
    for path in sorted(glob.glob("/home/zhuojun/prof/st2-%s-*.json" % tag)):
        try:
            d = json.load(open(path))
        except Exception as exc:
            print(tag, os.path.basename(path), "unreadable", exc); continue
        ch = d["choices"][0]; m = ch.get("message", {})
        c = m.get("content") or ""; r = m.get("reasoning_content") or ""
        t = d.get("timings", {}) or {}
        u = d.get("usage", {}) or {}
        zeros = max([len(x) for x in re.findall(r"0+", c)] or [0])
        dn = t.get("draft_n", 0); da = t.get("draft_n_accepted", 0)
        out.append(dict(file=os.path.basename(path), comp=u.get("completion_tokens"),
                        finish=ch.get("finish_reason"), content=len(c), reason=len(r), zeros=zeros,
                        tok_s=t.get("predicted_per_second", 0.0),
                        prompt_s=t.get("prompt_per_second", 0.0), dn=dn, da=da,
                        acc=(da * 100.0 / dn) if dn else 0.0))
    return out

for tag in ("full", "opt"):
    rr = rows(tag)
    print("== route %s n=%d" % (tag, len(rr)))
    for x in rr:
        print("   %-16s comp=%-5s finish=%-7s content=%-5d reason=%-5d longest0=%-4d tok_s=%6.2f prompt_s=%7.2f draft=%d acc=%d (%.1f%%)" % (
            x["file"], x["comp"], x["finish"], x["content"], x["reason"], x["zeros"],
            x["tok_s"], x["prompt_s"], x["dn"], x["da"], x["acc"]))
    if rr:
        print("   SUMMARY %s n=%d tok_s median=%.2f min=%.2f max=%.2f | content median=%.0f | acc median=%.1f%% | degenerate(<200)=%d" % (
            tag, len(rr), statistics.median(x["tok_s"] for x in rr),
            min(x["tok_s"] for x in rr), max(x["tok_s"] for x in rr),
            statistics.median(x["content"] for x in rr),
            statistics.median(x["acc"] for x in rr),
            sum(1 for x in rr if x["content"] < 200)))
PY
echo "STATS_DONE" >> "$LOG"
