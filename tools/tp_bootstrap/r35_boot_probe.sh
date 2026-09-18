#!/bin/bash
echo '--- boot list ---'
journalctl --list-boots 2>/dev/null | tail -6 || echo 'no journalctl'
echo '--- previous boot tail (shutdown reason) ---'
journalctl -b -1 -n 25 --no-pager 2>/dev/null || echo 'no previous boot journal'
echo '--- previous boot: shutdown-related ---'
journalctl -b -1 --no-pager 2>/dev/null | grep -iE 'shutdown|poweroff|reboot|System is|Stopped|OOM|killed process' | tail -20 || echo 'none'
echo '--- current boot first 12 ---'
journalctl -b -n 0 --no-pager 2>/dev/null | head -12 || true
echo '--- .wslconfig ---'
for f in /mnt/c/Users/zhuojun/.wslconfig /etc/wsl.conf; do
  echo "== $f"; cat "$f" 2>/dev/null || echo '(missing)'
done
echo '--- windows-side wsl processes (via /mnt/c) ---'
ls -l /mnt/c/Users/zhuojun/.wslconfig 2>/dev/null
echo '--- wsl boot time ---'
uptime -s; who -b 2>/dev/null
