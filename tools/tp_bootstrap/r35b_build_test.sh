#!/bin/bash
# Round 35b: rebuild then rerun the TP-2 correctness suite in one session.
set -u
echo '=== BUILD ==='
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/build_r35.sh
echo '=== TESTS ==='
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/run_r35_tests.sh
