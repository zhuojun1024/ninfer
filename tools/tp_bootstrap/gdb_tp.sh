#!/usr/bin/env bash
sudo -n apt-get install -y gdb > /tmp/gdb_install.log 2>&1
command -v gdb || { echo NO_GDB; tail -5 /tmp/gdb_install.log; exit 1; }
cd /home/zhuojun/ninfer || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
gdb -batch -ex run -ex bt --args ./build/tests/ninfer_tp_device_pair_test 2>&1 | tail -40
