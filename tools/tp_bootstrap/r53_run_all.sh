#!/bin/bash
# Round 53: sync + build, then the full verification suite.
set -u
TB=/mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap
echo "=== build ==="
bash "$TB/build_r35.sh" 2>&1 | tail -5
echo "=== suite ==="
bash "$TB/r53_final.sh"
echo "=== done ==="
