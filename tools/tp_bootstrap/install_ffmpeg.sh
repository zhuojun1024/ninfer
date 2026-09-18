#!/usr/bin/env bash
echo "=== current ffmpeg dev packages ==="
for p in libavformat-dev libavcodec-dev libavutil-dev libswscale-dev; do
  dpkg -l "$p" 2>/dev/null | grep -q "^ii" && echo "$p: installed" || echo "$p: MISSING"
done
echo "=== .pc files ==="
ls /usr/lib/x86_64-linux-gnu/pkgconfig/ 2>/dev/null | grep -E "libav|libsw" | head
echo "=== installing ==="
sudo -n apt-get install -y libavformat-dev libavcodec-dev libavutil-dev libswscale-dev 2>&1 | tail -5
echo "=== verify ==="
pkg-config --modversion libavformat libavcodec libavutil libswscale
