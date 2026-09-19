#!/bin/bash
# Decompose TP-2 resident memory by controlled configuration deltas.
set -u
PROF=/home/zhuojun/prof
MODEL=/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
LOG=$PROF/r51_memsplit.log
: > "$LOG"
pkill -9 -f 'serve_supervis[e]' 2>/dev/null || true

stop() {
  pkill -9 -f 'build/apps/ninfer-serv[e]' 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f 'build/apps/ninfer-serv[e]' >/dev/null || break; sleep 1; done
}

run_case() {
  local tag="$1"; shift
  stop
  sleep 2
  cd /home/zhuojun/ninfer
  local slog="$PROF/r51_$tag.log"
  : > "$slog"
  ./build/apps/ninfer-serve "$MODEL" --devices 0,1 --port 8088 "$@" > "$slog" 2>&1 &
  for _ in $(seq 1 160); do
    grep -aq 'listening on http://127.0.0.1:8088' "$slog" && break
    sleep 2
  done
  sleep 3
  echo "=== case $tag | $*" >> "$LOG"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >> "$LOG" 2>&1
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> "$LOG" 2>&1
  grep -a 'listening on' "$slog" | head -1 >> "$LOG"
  stop
  sleep 3
}

run_case base    --kv-dtype fp8 --max-context 2048   --kv-capacity 2048
run_case basemtp --kv-dtype fp8 --max-context 2048   --kv-capacity 2048 --spec mtp --draft-tokens 2
run_case kv      --kv-dtype fp8 --max-context 131072 --kv-capacity 131072
run_case mtp     --kv-dtype fp8 --max-context 131072 --kv-capacity 131072 --spec mtp --draft-tokens 2
run_case mtpopt  --kv-dtype fp8 --max-context 131072 --kv-capacity 131072 --spec mtp --draft-tokens 2 --lm-head-draft
echo "=== device list" >> "$LOG"
nvidia-smi -L >> "$LOG" 2>&1
echo MEMSPLIT_DONE >> "$LOG"
cat "$LOG"
