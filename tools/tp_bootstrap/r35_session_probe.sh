#!/bin/bash
echo '--- previous boot: session lifecycle ---'
journalctl -b -1 --no-pager 2>/dev/null | grep -iE 'New session|session-|logged out|Removed session|Started Session|wsl' | tail -25
echo '--- previous boot 23:53:00-23:53:28 ---'
journalctl -b -1 --no-pager --since '23:52:50' --until '23:53:30' 2>/dev/null | head -40
echo '--- previous boot: our serve mentions ---'
journalctl -b -1 --no-pager 2>/dev/null | grep -iE 'ninfer|ninfer-serve' | tail -10 || echo 'none'
