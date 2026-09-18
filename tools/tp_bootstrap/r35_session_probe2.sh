#!/bin/bash
echo '--- New session events in boot -1 ---'
journalctl -b -1 --no-pager 2>/dev/null | grep -iE 'New session [0-9]+ of user|session [0-9]+:|Created session|Start of session' | tail -20
echo '--- systemd-logind session info (current boot) ---'
loginctl list-sessions 2>/dev/null || echo 'no loginctl'
echo '--- who ---'
who -a 2>/dev/null | head
echo '--- what is on pts ---'
ps -eo pid,ppid,tty,stat,cmd --sort=start_time 2>/dev/null | grep -E 'pts|systemd' | head -20
