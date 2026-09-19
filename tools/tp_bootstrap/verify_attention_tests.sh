#!/bin/bash
# Attention-route regression after the small-T per-device opt-in change.
set -u
cd /home/zhuojun/ninfer/build || exit 1
for t in ninfer_softmax_attention_test ninfer_kv_cache_append_test ninfer_context_kv_materialize_test ninfer_sliding_window_attention_test; do
  if ./tests/"$t" > /tmp/"$t".log 2>&1; then echo "PASS $t"; else echo "FAIL $t"; tail -6 /tmp/"$t".log; fi
done
