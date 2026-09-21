#!/bin/bash
# Round 35/36 build: sync the edited source trees into the WSL build tree and build.
# Directory-level sync (not a file list): a per-file list silently misses a header and the failure
# looks like a source error in the WSL copy.
set -u
SRC=/mnt/d/Documents/workbench/ninfer
DST=/home/zhuojun/ninfer
if command -v rsync > /dev/null 2>&1; then
  for d in src tests apps include bench tools cmake third_party; do
    [ -d "$SRC/$d" ] || continue
    rsync -a "$SRC/$d/" "$DST/$d/" || { echo "SYNC FAILED $d"; exit 1; }
  done
  # The top-level build description carries platform branches too.
  for f in CMakeLists.txt; do
    [ -f "$SRC/$f" ] && rsync -a "$SRC/$f" "$DST/$f"
  done
else
  for d in src tests apps include bench; do
    [ -d "$SRC/$d" ] || continue
    (cd "$SRC" && find "$d" -type f -print0 | xargs -0 -I{} cp "{}" "$DST/{}") || {
      echo "SYNC FAILED $d"; exit 1; }
  done
fi
echo "--- synced $(date +%H:%M:%S) ---"
cd $DST
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake --build build -j 8 2>&1 | tail -60
echo "BUILD_EXIT=${PIPESTATUS[0]}"
