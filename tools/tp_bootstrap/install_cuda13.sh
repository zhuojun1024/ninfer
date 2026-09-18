#!/usr/bin/env bash
set -x
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb 2>&1 | tail -2
ls -la cuda-keyring.deb
sudo -n dpkg -i cuda-keyring.deb 2>&1 | tail -2
sudo -n apt-get update 2>&1 | tail -3
apt-cache search -n cuda-toolkit-13 2>/dev/null | head
apt-cache policy cuda-toolkit-13-1 2>/dev/null | head -5
