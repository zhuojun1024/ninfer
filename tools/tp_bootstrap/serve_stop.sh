#!/bin/bash
# Deliberately release both cards: stop the supervisor first (otherwise it relaunches the engine
# immediately and the next model load collides with it), then the engine itself.
set -u
pkill -9 -f 'serve_supervise.sh' 2>/dev/null
pkill -9 -f 'ninfer-serve' 2>/dev/null
for _ in $(seq 1 30); do
  pgrep -f 'ninfer-serve' > /dev/null || break
  sleep 1
done
pgrep -a -f 'ninfer-serve' && { echo 'REFUSING: ninfer-serve survived'; exit 1; }
echo 'ninfer-serve stopped'
