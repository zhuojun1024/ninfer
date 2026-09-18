#!/bin/bash
# Round 36b: sync, build the fold oracle, run it on a released pair of cards.
set -u
cd /mnt/d/Documents/workbench/ninfer && python3 tools/tp_bootstrap/patch4.py || exit 2
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
for d in tests src; do rsync -a "$SRC/$d/" "$DST/$d/" || exit 3; done
cd $DST
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build --target ninfer_gdn_replay_fold_test -j 8 2>&1 | tail -5
echo "BUILD_EXIT=${PIPESTATUS[0]}"
./build/tests/ninfer_gdn_replay_fold_test
echo "FOLD_TEST_EXIT=$?"
