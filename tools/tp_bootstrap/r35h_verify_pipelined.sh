#!/bin/bash
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/serve_stop.sh
echo '=== CORRECTNESS (pipelined AR) ==='
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/run_r35_tests.sh
echo '=== PIPELINED AR MEASUREMENT ==='
bash /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/r35h_pipelined_measure.sh
