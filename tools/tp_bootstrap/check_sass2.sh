#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
echo "=== SERVE fatbin sections ==="
cuobjdump --list-fatbin $B/apps/ninfer-serve 2>&1 | head -20
echo "=== SERVE total fatbin functions (grep Function) ==="
cuobjdump --list-fatbin $B/apps/ninfer-serve 2>/dev/null | grep -c "Function"
echo "=== TEST fatbin sections ==="
cuobjdump --list-fatbin $B/tests/ninfer_qwen3_5_tp2_forward_test 2>&1 | head -20
echo "=== TEST total fatbin functions ==="
cuobjdump --list-fatbin $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null | grep -c "Function"
