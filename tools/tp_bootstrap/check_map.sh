#!/bin/sh
SMI_IDX=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
echo "$SMI_IDX"
for d in 0 1; do
  UUID=$(awk -v i=$d '$1 == i {print $3}' /tmp/cuda_devlist.txt)
  SMI=$(echo "$SMI_IDX" | awk -F', ' -v u=$UUID '$2 == u {print $1}')
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $SMI)
  echo "CUDA $d -> nvidia-smi $SMI, used ${used} MiB"
done
