#!/bin/bash
# Round 53 verification: static shard split (MTP on shard 0, Vision on shard 1).
#   1. regressions with the cards free, including the new --vision shard assertions,
#   2. the shipped 262,144 service restarted WITH --vision (the memory verdict),
#   3. text greedy A/B against the pre-change reference (placement must stay numerically neutral),
#   4. real image requests.
set -u
BIN=/home/zhuojun/ninfer/build/tests
ART=/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
LOG=/home/zhuojun/prof/r53_final.log
: > "$LOG"
run() { echo "=== $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; echo "EXIT=$?" >> "$LOG"; }
bash "$TB/serve_stop.sh" >> "$LOG" 2>&1
run "$BIN/ninfer_qwen3_5_tp2_load_test" --artifact "$ART"
run "$BIN/ninfer_qwen3_5_tp2_load_test" --artifact "$ART" --spec mtp
run "$BIN/ninfer_qwen3_5_tp2_load_test" --artifact "$ART" --spec mtp --lm-head-draft
run "$BIN/ninfer_qwen3_5_tp2_load_test" --artifact "$ART" --spec mtp --vision
run "$BIN/ninfer_qwen3_5_tp2_load_test" --artifact "$ART" --vision
run "$BIN/ninfer_qwen3_5_tp2_forward_test" --artifact "$ART"
run "$BIN/ninfer_qwen3_5_vision_workspace_test"
run "$BIN/ninfer_qwen3_5_visual_scatter_test"
run "$BIN/ninfer_embedding_test"
run "$BIN/ninfer_linear_tp2_split_fp8_head_test"
run "$BIN/ninfer_linear_tp2_split_grouped_head_test"
run "$BIN/ninfer_linear_tp2_split_nvfp4_test"
run "$BIN/ninfer_tp_device_pair_test"
grep -E "===|passed|OK |FAIL|EXIT=" "$LOG"
echo "--- restarting the shipped service at 262144 with --vision ---"
setsid nohup bash "$TB/serve_supervise.sh" 262144 8088 '--kv-dtype fp8 --temperature 0.7 --top-k 20 --top-p 0.80 --spec mtp --draft-tokens 2 --lm-head-draft --vision' >/dev/null 2>&1 &
for i in $(seq 1 300); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8088/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "service healthy after ${i}s"; break; }
  sleep 1
done
echo "--- ledger ---"
grep '\[mem\]' /home/zhuojun/prof/serve_supervised.log | tail -2
echo "--- text greedy A/B (must match ab-embed) ---"
bash "$TB/r52_ab.sh" v53
python3 - <<'PY'
import json
def load(tag):
    return [json.loads(line) for line in open("/home/zhuojun/prof/ab-%s.jsonl" % tag, encoding="utf-8")]
ref = load("embed")
cur = load("v53")
ok = all(x.get("hash") == y.get("hash") for x, y in zip(ref, cur))
print("v53", "IDENTICAL to embed" if ok else "DIFFERS")
if not ok:
    for x, y in zip(ref, cur):
        if x.get("hash") != y.get("hash"):
            print("   req%d ref %s(%d) vs %s(%d)" % (x["i"], x["hash"], x["len"], y["hash"], y["len"]))
PY
echo "--- image requests ---"
python3 "$TB/r53_vision.py" 2>&1 | tail -40
echo "--- idle gpu ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
