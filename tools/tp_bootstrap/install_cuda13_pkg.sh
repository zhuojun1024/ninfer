#!/usr/bin/env bash
sudo -n apt-get install -y cuda-toolkit-13-1 > /tmp/cuda13_install.log 2>&1
rc=$?
echo INSTALL_EXIT=$rc
tail -3 /tmp/cuda13_install.log
ls /usr/local/ | grep cuda
/usr/local/cuda-13.1/bin/nvcc --version 2>/dev/null | tail -1
