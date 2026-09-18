#!/bin/bash
# Stage 36.1 verification: load the TP-2 shards with and without MTP, then bring the serve back.
set -u
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/build_r35.sh
echo '=== load test: no spec ==='
cd /home/zhuojun/ninfer
./build/tests/ninfer_qwen3_5_tp2_load_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer
echo "LOAD_PLAIN_EXIT=$?"
echo '=== load test: --spec mtp ==='
./build/tests/ninfer_qwen3_5_tp2_load_test --artifact /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer --spec mtp
echo "LOAD_MTP_EXIT=$?"
