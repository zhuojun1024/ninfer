#!/bin/sh
/usr/local/cuda-13.1/bin/nvcc -O2 -arch=sm_120a \
  /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/cuda_devlist.cu \
  -o /tmp/cuda_devlist -lcuda 2>&1
echo '--- CUDA enumeration ---'
/tmp/cuda_devlist
echo '--- detected DEVS ---'
/tmp/cuda_devlist | awk '$2 == "12.0" {print $1}' | head -2 | paste -sd, -
