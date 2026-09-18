#!/usr/bin/env bash
dpkg -l libcurl4-openssl-dev 2>/dev/null | grep -q "^ii" && echo "libcurl4-openssl-dev: installed" || { sudo -n apt-get install -y libcurl4-openssl-dev 2>&1 | tail -3; }
pkg-config --modversion libcurl
