#!/bin/bash
# Focused regression for the head-split GDN replay-record change.
set -u
cd /home/zhuojun/ninfer/build || exit 1
status=0
tests="tests/ninfer_gated_delta_net_replay_record_test tests/ninfer_gdn_replay_fold_test tests/ninfer_gdn_input_proj_conv_record_test tests/ninfer_gdn_input_proj_conv_snapshot_test tests/ninfer_gdn_input_proj_test"
extra=$(ls tests/ 2>/dev/null | grep -E 'tp2|device_pair' | sed 's|^|tests/|')
for t in $tests $extra; do
  if [ ! -x "$t" ]; then echo "SKIP $t (missing)"; continue; fi
  log=/tmp/$(basename "$t").log
  if ./"$t" > "$log" 2>&1; then
    echo "PASS $t"
  else
    code=$?
    echo "FAIL $t (exit $code)"
    tail -6 "$log"
    status=1
  fi
done
echo "FOCUSED_TESTS_EXIT=$status"
