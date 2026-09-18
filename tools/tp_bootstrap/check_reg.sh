#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
for BIN in apps/ninfer-serve tests/ninfer_qwen3_5_tp2_forward_test; do
  echo "########## $BIN ##########"
  echo "--- registration symbols (nm) ---"
  nm $B/$BIN 2>/dev/null | grep -iE "__cudaRegister|__nv_module_ids|__cudaRegisterAll|RegisterFatBinary|RegisterFunction" | head -20
  echo "(reg symbol count: $(nm $B/$BIN 2>/dev/null | grep -icE '__cudaRegister|__nv_module_ids'))"
  echo "--- nv/cuda sections (readelf -S) ---"
  readelf -S $B/$BIN 2>/dev/null | grep -iE "nv_|cuda|fatbin|module"
done
