#!/bin/bash
# Round 55 build: sync the edited source trees into the WSL build tree and build build_dyn.
# The WSL build directory is build_dyn (Ninja); build_r35.sh still points at the removed build/.
set -u
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
if command -v rsync > /dev/null 2>&1; then
  for d in src tests apps include bench tools cmake third_party; do
    [ -d "$SRC/$d" ] || continue
    rsync -a "$SRC/$d/" "$DST/$d/" || { echo "SYNC FAILED $d"; exit 1; }
  done
  for f in CMakeLists.txt; do
    [ -f "$SRC/$f" ] && rsync -a "$SRC/$f" "$DST/$f"
  done
fi
echo "--- synced $(date +%H:%M:%S) ---"
cd $DST
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build_dyn -j 8 2>&1 | tail -60
echo "BUILD_EXIT=${PIPESTATUS[0]}"
