#!/bin/bash
cd /home/zhuojun/ninfer
./build/tests/ninfer_qwen3_5_tp2_forward_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer > /tmp/tp2fwd.log 2>&1
echo "TEST_EXIT=$?"
echo "=== lines of interest ==="
grep -E 'linear-diag|FAIL|PASS|probes|shard state' /tmp/tp2fwd.log
