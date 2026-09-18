#!/bin/sh
NSYS=/usr/local/cuda-13.1/bin/nsys
$NSYS stats --report cuda_gpu_kern_sum --format csv /tmp/tp2_prof.nsys-rep 2>/dev/null | head -60
